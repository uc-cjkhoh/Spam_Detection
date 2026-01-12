import os
import sys
import mlflow
import argparse
import numpy as np
import pandas as pd 

from prefect import flow, task
from prefect.cache_policies import NO_CACHE 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE

from src.data_loader.preprocessing import get_normalized_messages 
from src.utils.util import setup_core_components, finish_labelling
 

@flow(name='Setup environment') 
def setup_environment():
    @task(name='Download first batch data', cache_policy=NO_CACHE)
    def download_initial_data(config, db, filepath, target_column):
        data = db.get_records(config.data.query, columns=config.data.column_name) 
        data[target_column] = None 
        data.to_excel(filepath, index=False) 
        
    config, database, metadata, embedding_model, vectorstore, model = setup_core_components()  
     
    # Download first batch of stratified sample in local
    if not os.path.exists(config.models.initial_data_filepath): 
        download_initial_data(config, database, config.models.initial_data_filepath, args.target_column)
        sys.exit('Please label data manually before proceeding')
    
    # Check if finish labelling first batch of data
    if not finish_labelling(config.models.initial_data_filepath, args.target_column):
        raise ValueError(f"Please finish labelling the data in {config.models.initial_data_filepath}")
    
    return config, database, metadata, embedding_model, vectorstore, model
         
@task(name='Initialize Model', cache_policy=NO_CACHE)
def initialize_model(config, db, embedding_model, model):
    x, y = get_train_data(config, db, embedding_model)
    train_models(model, x, y)      

@task(name='Spam classification', cache_policy=NO_CACHE)
def spam_classification(model, embeddings): 
    return model.predict(embeddings), model.predict_proba(embeddings).max(axis=1)

@task(name='Model training', cache_policy=NO_CACHE)
def train_models(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
       
    model_instance = model.fit(x_train, y_train, eval_set=[x_test, y_test])
       
    with mlflow.start_run(run_name='Build/Update Model'): 
        mlflow.log_param('model_parameters', model_instance.get_params())
        mlflow.sklearn.log_model(
            sk_model=model_instance,
            name=type(model_instance).__name__,
            registered_model_name=f'{type(model_instance).__name__}',
            input_example=x[:1]
        )
 
@flow(name='Data Preprocessing')
def preprocess_data(embedding_model, message_df, target_column):
    @task(name="Dimension reduction", cache_policy=NO_CACHE)
    def dimension_reduction(embeddings: np.ndarray):
        pca = PCA(n_components=384)
        scaled_embedding = pca.fit_transform(embeddings)
        return scaled_embedding

    @task(name="Normalize message", cache_policy=NO_CACHE)
    def normalize_message(data, target_column):
        return get_normalized_messages(data, target_column)
        
    @task(name="Sentence embeddings", cache_policy=NO_CACHE)
    def get_embeddings(normalized_message, embedding_model):    
        return np.asarray(embedding_model.embed_documents(normalized_message)) 
    
    normalized_message = normalize_message(message_df, target_column=target_column) 
    embeddings = get_embeddings(normalized_message, embedding_model)             
    scaled_embeddings = dimension_reduction(embeddings)
    
    return embeddings, scaled_embeddings
    
@flow(name='Load training data')
def get_train_data(config, db, embedding_model):
    @task(name='Oversampling', cache_policy=NO_CACHE)
    def oversampling(x, y):
        smote = SMOTE(random_state=42)
        resampled_x, resampled_y = smote.fit_resample(x, y)    
        return resampled_x, resampled_y 
    
    # combine labeled data
    initial_data = pd.read_excel(config.models.initial_data_filepath)
    pseudo_data =  db.get_records(config.pseudo_query, columns=initial_data.columns)
    human_data = db.get_records(config.human_query, columns=initial_data.columns)
    train_data = pd.concat([initial_data, pseudo_data, human_data])
    
    train_data = train_data.fillna('')
      
    _, scaled_embeddings = preprocess_data(embedding_model, train_data, target_column=config.data.target_column)
      
    labels = train_data.loc[:, args.target_column].astype(int)

    x, y = oversampling(scaled_embeddings, labels)
            
    return x, y 
 
@flow(name='Active Learning Pipeline')
def main(args):   
    try:
        # Setup necessary components
        config, database, _, embedding_model, vectorstore, model = setup_environment()  
        
        # skip the first train_models when module start if model exists
        check_is_fitted(model.model)
        
        while True:
            # get training data
            x, y = get_train_data(config, database, embedding_model)
            
            # update model
            train_models(model, x, y)
            
            # select stratified sample in this day, group by hour
            data = database.get_records(config.data.query, columns=config.data.column_name)
    
            # preprocess data
            _, scaled_embeddings = preprocess_data(embedding_model, data, target_column=config.data.target_column)
            
            # classification
            result, confidence_score = model.predict(scaled_embeddings), model.predict_proba(scaled_embeddings)
            
            # label them by confidence score
            high_conf_ids = np.where(confidence_score >= args.threshold)[0]
            uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), args.number_of_uncertain)[:args.number_of_uncertain]
            label_status = np.zeros(confidence_score.shape)
            label_status[high_conf_ids] = 1
            label_status[uncertain_ids] = -1
            
            # reset latch_batch value before update
            database.run_statement('UPDATE sms_spam_cd.metadata_result SET last_batch = False')
            
            # save result to mysql
            database.save_to_mysql(
                data=pd.DataFrame({
                    'id': data['id'],
                    'datetime': data['datetime'],
                    'spam_label': result,
                    'confidence_score': confidence_score,
                    'label_status': label_status,
                    'model': type(model).__name__,
                    'last_batch': [True] * len(data)
                }).to_dict(orient='records')
            )
            
            # allow user to label without terminate module
            print('Temporary pause for data checking ...')
            user_input = input('Press any key to process or `q` to quit ...')
            if user_input == 'q':
                break
    except:
        initialize_model(config, database, embedding_model, model)
    finally:
        database.close_connection() 
               
               
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument('-u', '--mlflow_uri', type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument('-e', '--experiment', type=str, default='SMS SPAM DETECTION', help='name of the experiment in mlflow')
    p.add_argument('-c', '--target_column', type=str, default='spam_label', help='the column in database that indicate the type of sms (spam or ham)')
    p.add_argument('-s', '--skip_initialization', type=bool, default=True, help='whether to skip model initialization')
    p.add_argument('-n', '--number_of_uncertain', type=int, default=2800, help='configure the number of uncertain message for human label')
    p.add_argument('-t', '--threshold', type=float, default=0.975, help='configure the confidence score threshold')
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)