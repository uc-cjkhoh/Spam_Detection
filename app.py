import pandas as pd
import numpy as np
import faiss
import os 

from prefect import flow, task
from sklearn.linear_model import SGDClassifier
 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS

from src.config_folder.config_loader import get_config
from src.data_loader.data_loader import Database
from src.data_loader.preprocessing import text_normalize
from src.utils.util import create_required_folder_file


@task
def setup_environment(config: dict): 
    database = Database(config)
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.models.text_embedding.model_name,
        model_kwargs={"trust_remote_code": True}
    )
  
    return database, embedding_model 

@task
def initialize_vectorstore(config: dict):
    default_file_location = os.path.join(config.vectorstore.directory, config.vectorstore.filename)
    if os.path.exists(default_file_location):
        return FAISS.load_local(default_file_location)
    else:
        return None


@flow(name='SMS_SPAM_DETECTION')
def main():
    try:
        # get environment config
        config = get_config() 
        
        # create required directory
        create_required_folder_file(config)
        
        # setup environment
        mysql, embedding_model = setup_environment(config)
        
        # setup vectorstore
        loaded_index = initialize_vectorstore(config)
        
        # get subdata's metadata
        metadata = mysql.retrieve_by_query(config.metadata.query, columns=config.metadata.column_name)
        
        # start active learning process
        for i in range(len(metadata)):
            # 1. edit subdata's query
            data_query = config.data.query.format(*metadata.iloc[i])
            
            # 2. execute query & get subdata
            data = mysql.retrieve_by_query(data_query, columns=config.data.column_name)[:100]
            
            # 3. perform preprocessing
            data = text_normalize(data, target_column=config.data.target_column)
          
            # 4. save texts into vector database
            if loaded_index is not None:
                loaded_index.add_texts(
                    texts=data[config.data.target_column].tolist(), 
                    metadatas=[metadata.iloc[i].to_list()]
                )
            else:
                loaded_index = FAISS.from_texts(
                    texts=data[config.data.target_column].tolist(), 
                    embedding=embedding_model,
                    metadatas=[metadata.iloc[i].to_list()]
                )
            
            similar_text = loaded_index.similarity_search(query='you win a prize', k=5)
            print(similar_text)
            
            break
                     
        mysql.close_connection()
        vectorstore.close()
    except Exception as e:
        raise Exception(e) 
    
if __name__ == '__main__': 
    main()