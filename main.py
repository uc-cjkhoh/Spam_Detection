import os
import mlflow
import numpy as np
import pandas as pd 

from mlflow.exceptions import MlflowException

from sklearn.preprocessing import StandardScaler
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from langchain_huggingface import HuggingFaceEmbeddings

from src.config_folder.config_loader import get_config
from src.data_loader.database import Database
from src.data_loader.preprocessing import feature_engineering
from src.vector_database.vectorstore import VectorStore 
from src.ml.model_training import SGD 
from src.utils.util import create_require_files_and_directories, get_unique_pattern_ids, \
    stratified_sampling, oversampling
  
 
@task(name='Setup Environment', cache_policy=NO_CACHE)
def setup_environment(config) -> tuple[Database, VectorStore, HuggingFaceEmbeddings, SGD]:
    """Setup environment by create core components

    Args:
        config (dict): loaded configuration file (config.yaml)
        
    Raises:
        KeyError: when accessing unknown configuration key
        ValueError: when parameter value is wrong
        TypeError: when parameter in wrong type, wrong name or missing parameter
        MLFlowException: any unexpected exception happened when connect to mlflow

    Returns:
        tuple[Database, VectorStore, HuggingFaceEmbeddings, SGD]: all core components
    """
    
    logger = get_run_logger()
    
    try:
        logger.info('Connect to MLFlow')
        mlflow.set_tracking_uri(config.mlflow_uri)
        mlflow.set_experiment(config.experiment_name)
        
        logger.info('Create files and directories')
        create_require_files_and_directories(config)   
        
        logger.info('Create embedding model')
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding.model_name, 
            model_kwargs={'trust_remote_code': True}, 
            encode_kwargs={
                'batch_size': config.embedding.batch_size, 
                'normalize_embeddings': config.embedding.normalize_embeddings
            }, 
            show_progress=config.embedding.show_progress
        )
        
        logger.info('Create SGD model')
        model = SGD(model_name=config.ml_model.model_name) 
        
        
        logger.info('Connect to MySQL')
        database = Database(
            host=config.database.host, 
            port=config.database.port, 
            user=config.database.user, 
            password=config.database.password,
            schema=config.database.schema
        )
            
        logger.info('Create vectorstore')
        vectorstore = VectorStore(
            directory=config.vectorstore.directory, 
            filename=f'sms_embeddings_{str.lower(config.experiment_name)}',
            embedding=embedding_model
        )
        
        if f'sms_embeddings_{str.lower(config.experiment_name)}' + '.pkl' in os.listdir(config.vectorstore.directory):
            logger.info('Load index from saved file')
            vectorstore.load_local_file()
        else:
            logger.info('Get initial data') 
            initial_data = database.get_records(config.initial_data)
            
            logger.info('Initialize vectorstore with initial data')
            vectorstore.load_with_dataframe(initial_data)
        
    except KeyError:
        logger.error('No such key in config', exc_info=True)
        raise
    except MlflowException:
        logger.error('Error happened when trying to setup mlflow', exc_info=True)
        raise
    except (ValueError, TypeError) as e:
        logger.error(f'Failed to setup environment due to {e}', exc_info=True)
        raise
    
    return database, vectorstore, embedding_model, model
        

@task(name='Load Training Data', cache_policy=NO_CACHE)
def load_training_data(config, database, vectorstore, embedding_model, scaler) -> tuple[np.ndarray, np.ndarray]:
    """Load training data

    Args:
        config (dict): configuration in config.yaml
        database (Database): MySQL database connection
        vectorstore (VectorStore): vectorstore entity
        embedding_model (HuggingFaceEmbeddings): embedding model

    Returns:
        tuple[np.ndarray, np.ndarray]: x_train, y_train
    """
    
    logger = get_run_logger()
    
    try:
        logger.info('Retrieve initial data from vectorstore')
        initial_payloads, initial_embeddings, initial_labels = vectorstore.return_pagecontent_embeddings_and_labels()
        
        logger.info('Perform feature engineering')
        initial_features = feature_engineering(initial_payloads) 
        
        stds = initial_features.std(axis=0)
        non_variance_col_ids = stds > 0
        initial_features = initial_features.loc[:, non_variance_col_ids]
        
        initial_features = scaler.fit_transform(initial_features)
        
        logger.info('Stack embeddings and features horizontally')
        combined_initial_embeddings = np.hstack((initial_features, initial_embeddings)) 
        
        x_train = combined_initial_embeddings
        y_train = initial_labels
        
        logger.info('Load labeled data in MySQL')
        prelabeled_data = database.get_records(config.labeled_data)
        
        if len(prelabeled_data) > 0:
            prelabeled_payloads = prelabeled_data[config.target_column]
            prelabeled_labels = prelabeled_data[config.label_column]
            
            logger.info('Perform feature engineering')
            prelabeled_features = feature_engineering(prelabeled_payloads) 
            prelabeled_features = prelabeled_features.loc[:, non_variance_col_ids]
            
            prelabeled_features = scaler.transform(prelabeled_features)
            
            logger.info('Perform sentence embeddings')
            prelabeled_embeddings = embedding_model.embed_documents(prelabeled_payloads)
            
            logger.info('Stack embeddings and features horizontally')
            combined_prelabeled_embeddings = np.hstack((prelabeled_features, prelabeled_embeddings))
            
            logger.info('Stack initial data and labeled data vertically')
            x_train = np.vstack((combined_initial_embeddings, combined_prelabeled_embeddings))
            y_train = np.concatenate((np.ravel(initial_labels), np.ravel(prelabeled_labels))).astype(int)
        
        retent_ids = get_unique_pattern_ids(x_train)
        x_train = x_train[retent_ids]
        y_train = y_train[retent_ids]
        logger.info(f'Reduce number of payload to {len(x_train)}')
        
        logger.info('Perform oversampling')
        x_train, y_train = oversampling(x_train, y_train) 
        
        return x_train, y_train, non_variance_col_ids
 
    except KeyError as e:
        logger.error(f'Missing config key: {e}', exc_info=True)
        raise
    except TypeError as e:
        logger.error(f'Invalid paramter: {e}', exc_info=True)
        raise
    except AttributeError as e:
        logger.error(f'Invalid method: {e}', exc_info=True)
        raise
    except ValueError as e:
        logger.error(f'Invalid data shape or value: {e}', exc_info=True)
        raise


@task(name='Load Testing Data', cache_policy=NO_CACHE)
def load_testing_data(config, database, embedding_model, scaler, zero_variance_column_ids) -> tuple[np.ndarray, np.ndarray]:
    """Load testing data

    Args:
        config (dict): configuration from config.yaml
        database (Database): MySQL database connection
        embedding_model (HuggingFaceEmbeddings): embedding_model

    Returns:
        tuple[np.ndarray, np.ndarray]: x_test, y_test
    """
    
    logger = get_run_logger()
    
    try:
        test_data = database.get_records(config.test_data) 
        test_payloads = test_data[config.target_column]
        test_labels = test_data[config.label_column]
        
        test_features = feature_engineering(test_payloads.copy())
        test_features = test_features.loc[:, zero_variance_column_ids]
        test_features = scaler.transform(test_features)
        
        test_embeddings = embedding_model.embed_documents(test_payloads)
        combined_test_embeddings = np.hstack((test_features, test_embeddings))
        
        return combined_test_embeddings, test_labels
    
    except KeyError as e:
        logger.error(f'Invalid config key: {e}', exc_info=True)
        raise
    except TypeError as e:
        logger.error(f'Invalid parameter: {e}', exc_info=True)
        raise
    

@flow(name='Active Learning Pipeline')
def main():    
    logger = get_run_logger()
    scaler = StandardScaler()
    
    logger.info('Retrieve configuration')
    config = get_config() 
    
    logger.info('Setup environment')
    database, vectorstore, embedding_model, model = setup_environment(config)
    
    logger.info('Load training data')
    x_train, y_train, zero_variance_column_ids = load_training_data(config, database, vectorstore, embedding_model, scaler)
    
    logger.info('Load testing data') 
    x_test, y_test = load_testing_data(config, database, embedding_model, scaler, zero_variance_column_ids)

    logger.info('Train model')
    model.fit(x_train, y_train)

    logger.info('Evaluate model')
    model.evaluate(x_test, y_test)
    
    # for i in range(5):
    #     # stratified sampling
    #     new_batch_data_id, new_batch_data_dt, new_batch_data_msg = stratified_sampling(config, database)

    #     # preprocess data
    #     scaled_embeddings = preprocess_data(embedding_model, new_batch_data_msg.copy(), target_column=config.target_column)
        
    #     # # reduce duplicate pattern
    #     # unique_pattern_id = get_unique_pattern_ids(scaled_embeddings)
    #     # scaled_embeddings = scaled_embeddings[unique_pattern_id]
        
    #     # get classification and confidence score
    #     result = model.predict(scaled_embeddings)
    #     confidence_score = model.predict_proba(scaled_embeddings)
        
    #     # get high confidence and uncertain data indexes
    #     high_conf_ids = np.where(confidence_score >= args.threshold)[0]
    #     uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), args.number_of_uncertain)[:args.number_of_uncertain]
        
    #     # create new training data (initial data + high confidence data)
    #     high_conf_embeddings = scaled_embeddings[high_conf_ids]
    #     high_conf_labels = result[high_conf_ids].squeeze()
        
    #     # label based on vectorstore
    #     uncertain_embeddings = scaled_embeddings[uncertain_ids]
    #     labels_status_by_vectordb, labels_by_vectordb = vectorstore.label_uncertains(new_batch_data_msg.iloc[uncertain_ids].iloc[:, 0].to_list())
        
    #     # configure label status (-1 = need human to label)
    #     label_status = np.zeros(confidence_score.shape)
    #     label_status[high_conf_ids] = 1
    #     label_status[uncertain_ids] = labels_status_by_vectordb
        
    #     # replace teacher model's uncertain result with labels made by vectorstore
    #     result[uncertain_ids] = labels_by_vectordb
        
    #     # combine data
    #     x_train, y_train = np.vstack((x_train, high_conf_embeddings, uncertain_embeddings)), np.hstack((y_train, high_conf_labels, labels_by_vectordb))
        
    #     # oversampling it 
    #     x_train, y_train = oversampling(x_train, y_train)
        
    #     # train model model
    #     model.fit(x_train, y_train.reshape(-1, 1))

    #     # evaluate and save model model
    #     model.evaluate(x_test, y_test)

    #     # save result to mysql
    #     database.save_to_mysql(
    #         pd.DataFrame({
    #             'id': new_batch_data_id,
    #             'datetime': new_batch_data_dt,
    #             'spam_label': result,
    #             'confidence_score': confidence_score,
    #             'label_status': label_status,
    #             'model': type(model).__name__
    #         }).to_dict(orient='records')
    #     )
        
    #     # update last_batch column
    #     database.run_statement(f'update sms_spam_cd.label_by_vectordb set iter_involved = case when label_status in (-1, 1) then concat(coalesce(iter_involved, ""), "1") else concat(coalesce(iter_involved, ""), "0") end')
         
    database.close_connection() 


if __name__ == '__main__':  
    main()