import pandas as pd
import numpy as np 
import argparse
import mlflow 

from prefect import flow, task
from prefect.cache_policies import NO_CACHE 
from langchain_huggingface import HuggingFaceEmbeddings 

# custom libraries  
from src.model.model_training import train_model, load_model
from src.data_loader.data_loader import Database
from src.data_loader.preprocessing import text_normalize 
from src.vector_database.vectorstore import VectorStore
from src.config_folder.config_loader import get_config
from src.utils.util import create_required_folder_file, update_metadata 


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
    
    
@task(cache_policy=NO_CACHE)
def text_preprocessing(data: pd.DataFrame, embedding_model, target_column: str):
    data = text_normalize(data, target_column=target_column)
    message_list = data[target_column].to_list() 
    embeddings = np.asarray(embedding_model.embed_documents(message_list))
    
    return embeddings, zip(message_list, embeddings)  


@flow(name='SMS_SPAM_DETECTION')
def main(args):  
    config = get_config() 
    create_required_folder_file(config) 
    mysql, embedding_model, vectorstore = setup_environment(config)
        
    metadata = mysql.get_population_metadata(config.metadata.query, columns=config.metadata.column_name) 
    update_metadata(config, metadata)
        
    for i in range(len(metadata)): 
        # get data
        data_query = config.data.query.format(*metadata.iloc[i]) 
        data, data_metadata = mysql.retrieve_subdata_by_query(data_query, columns=config.data.column_name)
        embeddings, text_embedding_pair = text_preprocessing(data, embedding_model, target_column=config.data.target_column)
         
        # save data to vectorstore
        vectorstore.write_to_vectorstore(text_embedding_pair, embedding_model, data_metadata) 
        
        # get prediction from model
        ml_model = load_model(config, mysql.get_cursor()) 
        y_pred, confidence_score = ml_model.predict(embeddings), ml_model.predict_proba(embeddings).max(axis=1)
        
        # train model with high confidence labels
        high_conf_idx = np.where(confidence_score > config.models.confidence_score_threshold)
        train_model(
            model=ml_model, 
            x=embeddings[high_conf_idx], 
            y=y_pred[high_conf_idx], 
            metadata=str(metadata)
        )
    
        update_metadata(config)
                    
    mysql.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection with SGDClassifier')
    p.add_argument("--mlflow_uri", type=str, default='file:./mlruns', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument("--model_id", type=str, default=None, help='specify trained model, else us new model') 
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)