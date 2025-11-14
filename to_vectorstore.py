import pandas as pd
import numpy as np 
import argparse
import mlflow 

from prefect import flow, task
from prefect.cache_policies import NO_CACHE 
from langchain_huggingface import HuggingFaceEmbeddings 

# custom libraries   
from data_loader.connection import Database
from src.data_loader.preprocessing import text_normalize 
from src.vector_database.vectorstore import VectorStore
from src.config_folder.config_loader import get_config
from src.utils.util import create_required_folder_file, update_metadata, generate_metadata


@task(cache_policy=NO_CACHE)  
def setup_environment(config: dict): 
    database = Database(config)
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.models.text_embedding.model_name,
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={
            'normalize_embeddings': True,
            'max_length': 1024,
            'batch_size': 4
        },
        show_progress=True
    )
    vectorstore = VectorStore(config.vectorstore) 
    return database, embedding_model, vectorstore   
      

@flow(name='Ingestion to Vectorstore')
def main():  
    config = get_config() 
    
    create_required_folder_file(config) 
    
    db, embedding_model, vectorstore = setup_environment(config)    
    
    metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
    
    update_metadata(config, metadata)
    
    for i in range(len(metadata)):  
        # modify query based on current metadata
        data_query = config.data.query.format(*metadata.iloc[i]) 
        
        # get message from mysql
        data = db.run_query(data_query, columns=config.data.column_name) 
         
        # normalize text
        data = text_normalize(data, target_column=config.data.target_column)
        
        # convert dataframe to list
        messages = data[config.data.target_column].to_list()
         
        # perform text embedding
        embeddings = np.asarray(embedding_model.embed_documents(messages))
        
        # generate metadata for each row
        data_metadata = pd.concat([
            data[config.data.metadata_column],
            pd.DataFrame({
                'label': [None] * len(data),
                'confidence_score': [None] * len(data),
                'label_status': ['unlabeled'] * len(data)
            })
        ], axis=1)
        
        # write embeddings into faiss
        vectorstore.write_to_vectorstore(zip(messages, embeddings), embedding_model, data_metadata) 
         
        # save progress status
        update_metadata(config)
                    
    db.close_connection()  
    
    
if __name__ == '__main__':  
    main()