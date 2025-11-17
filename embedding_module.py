import pandas as pd
import numpy as np

from prefect import flow, task

from src.data_loader.preprocessing import get_normalized_messages  
from src.config_folder.config_loader import get_config
from src.utils.util import setup_environment, create_required_folder_file, update_metadata, initialize_metadata
 
 
@task
def embed_messages(embedding_model, messages: list):
    return np.asarray(embedding_model.embed_documents(messages))


@flow(name='SMS_SPAM_DETECTION')
def main():  
    try:
        config = get_config() 
        
        create_required_folder_file(config) 
        
        db, embedding_model, vectorstore = setup_environment(config)    
        
        metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
        
        update_metadata(config, metadata)
        
        for i in range(len(metadata)):  
            data_query = config.data.query.format(*metadata.iloc[i]) 
            
            data = db.run_query(data_query, columns=config.data.column_name) 
            
            messages = get_normalized_messages(data, target_column=config.data.target_column) 
            
            embeddings = embed_messages(embedding_model, messages)
            
            data_metadata = initialize_metadata(data[config.data.metadata_column])
             
            vectorstore.write_to_vectorstore(zip(messages, embeddings), embedding_model, data_metadata) 
            
            update_metadata(config)
    except Exception as e:
        raise Exception(e)
    finally:            
        db.close_connection()  
    
    
if __name__ == '__main__': 
    main()