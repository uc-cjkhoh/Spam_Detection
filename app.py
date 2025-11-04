import pandas as pd
import numpy as np
import os
import mlflow 

from mlflow.models import infer_signature
from prefect import flow, task
from sklearn.linear_model import SGDClassifier

# ... require unit testing before import custom libraries

from testing.utils.util import check_exist_model
from testing.config_loader.config_loader import get_config
from testing.data_loader.data_loader import Database
from testing.data_loader.preprocessing import PreprocessPipeline
from testing.data_loader.embedding import EmbeddingPipeline
from testing.vector_database.vectorstore import VectorStore
from testing.model.ml_pipeline import MLPipeline


@task
def run_query(config, query, columns=None) -> pd.DataFrame:
    db = Database(config)  
    data = db.retrieve_by_query(query)
    
    if columns is not None:
        data.columns = columns
    
    db.close_connection()
    return data

@task
def preprocess(data: pd.Series) -> pd.Series: 
    preprocess = PreprocessPipeline()
    return preprocess.text_normalize(data)

@task
def text_embedding(model_name: str, data: pd.Series) -> np.ndarray:
    embeder = EmbeddingPipeline(model_name=model_name)
    return embeder.embed_message(data)

@task
def setup_vectorstore():
    vectorstore = VectorStore()
    if vectorstore.get_vectors() is None and os.listdir(vectorstore.get_vectorstore_filepath()) > 0:
        vectorstore.load_exist_vectorstore()
        
    return vectorstore

@task
def setup_model(config):
    has_exist_model = (os.listdir(config.models.save_model_to.folder) > 0)
    if has_exist_model:
        
    

@flow(name='SMS_SPAM_DETECTION')
def main():
    try:
        # setup
        config = get_config() 
        
        vectorstore = VectorStore()
        
        model = setup_model(config)
        
        # main process start here
        
        metadata = run_query(config, config.metadata.query)
         
        # ... require data quality check
        
        for i in range(metadata):
            data_query = config.data.query.format(*metadata.iloc[i])
            
            data = run_query(config, data_query, columns=config.data.column_name)
            
            data[config.data.target_column] = preprocess(data[config.data.target_column])
        
            embeddings = text_embedding(config.models.text_embedding.model_name, data[config.data.target_column])
            
            vectorstore.write(embeddings)
            
            with mlflow.start_run():   
                pass
        
        
    except Exception as e:
        raise Exception(e)    
    finally:
        vectorstore.close()
    
    
if __name__ == '__main__':
    mlflow.set_tracking_url(uri='http://127.0.0.1:4201')
    main()