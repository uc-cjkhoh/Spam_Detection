import os
import joblib
import numpy as np
 
from langchain_huggingface import HuggingFaceEmbeddings
 
from prefect import task, flow, get_run_logger
from prefect.cache_policies import NO_CACHE

from src.data_loader.database import Database
from src.vector_database.vectorstore import VectorStore

from src.utils.util import get_unique_pattern_ids
from src.config_folder.config_loader import get_config
from src.data_loader.preprocessing import feature_engineering

from data_validation.configs_validation.validate_config_loader import ProjectConfig
from data_validation.vectorstore_validation.validate_vectorstore import VectorstoreConfig

 
@task(name='Setup Environment', cache_policy=NO_CACHE)
def setup_environment(config: ProjectConfig) -> tuple[Database, HuggingFaceEmbeddings, VectorStore]:
    """Setup environment by create core components

    Args:
        config (ProjectConfig): loaded configuration file (config.yaml)
        
    Raises:
        KeyError: when accessing unknown configuration key
        ValueError: when parameter value is wrong
        TypeError: when parameter in wrong type, wrong name or missing parameter  

    Returns:
        tuple[Database, HuggingFaceEmbeddings, VectorStore]: all core components
    """
    
    logger = get_run_logger()
    
    try: 
        logger.info('Create embedding model')
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding.model_name,
            encode_kwargs={
                'batch_size': config.embedding.batch_size, 
                'normalize_embeddings': config.embedding.normalize_embeddings
            }, 
            show_progress=config.embedding.show_progress
        ) 
        
        logger.info('Connect to MySQL')
        database = Database( 
            host=config.database.host, 
            port=config.database.port, 
            username=config.database.user, 
            password=config.database.password,
            table_schema=config.database.table_schema 
        )
        
        logger.info('Create vectorstore')
        vectorstore = VectorStore(
            vectorstore_config=VectorstoreConfig(
                directory=config.vectorstore.directory, 
                filename=f'sms_embeddings_{str.lower(config.experiment_name)}'
            ),
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
        
        return database, embedding_model, vectorstore
        
    except KeyError:
        logger.error('No such key in config', exc_info=True)
        raise 
    except (ValueError, TypeError) as e:
        logger.error(f'Failed to setup environment due to {e}', exc_info=True)
        raise


@flow(name='Setup Initial Data')
def main():
    logger = get_run_logger()
    
    logger.info('Load configuration')
    config = get_config()
    
    scaler = joblib.load(f'./deploy/app/scaler/{config.experiment_name}_standard_scaler.joblib')
    
    logger.info('Get require components')
    database, embedding_model, vectorstore = setup_environment(config)
    
    logger.info('Retrieve data from MySQL')
    data = database.get_records(
        """
            with sample as (
                select 
                    id,
                    current_datetime as datetime,
                    payload,
                    smsc_src_addr,
                    smsc_dst_addr,
                    spam_filter_id,
                    row_number() over(partition by day(current_datetime), hour(current_datetime) order by rand()) as rn
                from
                    sms_spam_cd.data_tdr_spam_filter
                where
                    year(current_datetime) = 2026
                    and month(current_datetime) = 2
                    and (day(current_datetime) between 1 and 6)
                    and (hour(current_datetime) between 9 and 13)
            )
            select
                id,
                datetime,
                payload,
                smsc_src_addr,
                smsc_dst_addr,
                spam_filter_id 
            from
                sample
            where
                rn <= 300
        """
    )
    
    logger.info('Feature engineering')
    features = feature_engineering(data[config.target_column])
    features = scaler.transform(features)
    
    logger.info('convert payloads to embeddings')
    embeddings = embedding_model.embed_documents(data[config.target_column].to_list())
    
    final_embeddings = np.hstack((features, embeddings))
     
    logger.info('select only 5 embeddings from each cluster')
    retent_ids = get_unique_pattern_ids(final_embeddings, keep_n=5) 
    
    data = data.iloc[retent_ids, :]
    embeddings = np.asarray(embeddings)[retent_ids]
    
    _, labels = vectorstore.label_uncertains(embeddings, similarity_threshold=config.similarity_threshold)
    data['spam_label'] = labels
      
    logger.info('save to mysql')
    database.save_to_mysql(
        data=data,
        destination_table='new_initial_data'
    )
     
    database.close_connection()
    

if __name__ == '__main__':
    main()