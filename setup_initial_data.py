from langchain_huggingface import HuggingFaceEmbeddings
 
from prefect import task, flow, get_run_logger
from prefect.cache_policies import NO_CACHE

from src.config_folder.config_loader import get_config
from src.data_loader.database import Database
from src.vector_database.vectorstore import VectorStore 
from src.utils.util import get_unique_pattern_ids

from data_validation.configs_validation.validate_config_loader import ProjectConfig
from data_validation.vectorstore_validation.validate_vectorstore import VectorstoreConfig

 
@task(name='Setup Environment', cache_policy=NO_CACHE)
def setup_environment(config: ProjectConfig) -> tuple[Database, VectorStore, HuggingFaceEmbeddings]:
    """Setup environment by create core components

    Args:
        config (ProjectConfig): loaded configuration file (config.yaml)
        
    Raises:
        KeyError: when accessing unknown configuration key
        ValueError: when parameter value is wrong
        TypeError: when parameter in wrong type, wrong name or missing parameter  

    Returns:
        tuple[Database, VectorStore, HuggingFaceEmbeddings]: all core components
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
    
    logger.info('Get require components')
    database, embedding_model, vectorstore = setup_environment(config)
    
    logger.info('Retrieve data from MySQL')
    data = database.get_records(
        """
            with sample as (
                select 
                    id,
                    current_datetime,
                    payload,
                    smsc_src_addr,
                    smsc_dst_addr,
                    spam_filter_id,
                    seq_id
                    row_number over(partition by day(current_datetime), hour(current_datetime) order by rand()) as rn
                from
                    sms_spam_cd.data_tdr_spam_filter
                where
                    year(current_datetime) = 2026
                    and month(current_datetime) = 2
                    and (day(current_datetime) in between 1 and 6
            )
            
            select
                id,
                current_datetime,
                payload,
                smsc_src_addr,
                smsc_dst_addr,
                spam_filter_id,
                seq_id
            from
                sample
            where
                rn <= 300
        """
    )
    
    logger.info('convert payloads to embeddings')
    embeddings = embedding_model.embed_documents(data[config.target_column].to_list())
    
    # 3. cluster them 
    # hdbscan = HDBSCAN()
    
    logger.info('select only 10 embeddings from each cluster')
    retent_ids = get_unique_pattern_ids(embeddings, keep_n=10) 
    
    # 5. pre-label them if any of them has more or equal 0.9 in cosine similarity
    #    with the nearest embedding in previous vectordatabase
    
    data = data.iloc[retent_ids, :]
    data['spam_label'] = ""
    
    logger.info('save to mysql')
    database.save_to_mysql(
        data=data.to_dict(orient='records'),
        destination_table='initial_data'
    )
     

if __name__ == '__main__':
    main()