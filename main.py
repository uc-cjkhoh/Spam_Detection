import numpy as np
import os
import pandas as pd
import argparse
import json
import mlflow

from tqdm import tqdm
from prefect import flow, task 
from sklearn.decomposition import PCA

from src.data_loader.preprocessing import get_normalized_messages  
from src.config_folder.config_loader import get_config
from src.utils.util import setup_core_instances, create_required_folder_file, update_metadata, generate_metadata
from src.ml.model_training import train_model, load_model 
 
 
@task
def embed_messages(embedding_model, messages: list):
    return np.asarray(embedding_model.embed_documents(messages))
 
  
@task
def dimension_reduction(embeddings: np.ndarray):
    pca = PCA(n_components=min(embeddings.shape[0], embeddings.shape[-1]))
    return pca.fit_transform(embeddings)
 

@flow(name='Setup Environment') 
def setup_environment():
    config = get_config() 
    create_required_folder_file(config) 
    db, embedding_model, vectorstore = setup_core_instances(config)
    metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
    update_metadata(config, metadata)
    return config, db, embedding_model, vectorstore, metadata
    

@flow(name="Get Messages And Embeddings")
def get_normalized_messages_embeddings(config, data, embedding_model, data_query):   
    messages = get_normalized_messages(data, target_column=config.data.target_column)
    embeddings = embed_messages(embedding_model, messages)    
    return messages, embeddings


@flow(name="SMS Spam Detection")
def perform_ml_operations(config, db, embeddings): 
    ml_model = load_model(config, db.get_cursor())  
    y_pred, confidence_score = ml_model.predict(embeddings), ml_model.predict_proba(embeddings).max(axis=1) 
    return y_pred, confidence_score
 

@flow(name='Main')
def main(args):  
    try:
        config, db, embedding_model, vectorstore, metadata = setup_environment()
        
        for i in range(len(metadata)):   
            # format sql query
            data_query = config.data.query.format(*metadata.iloc[i])   
            
            # get messages
            data = db.run_query(data_query, columns=config.data.column_name) 
            
            # preprocess messages and embeddings
            normalized_messages, embeddings = get_normalized_messages_embeddings(config, data, embedding_model, data_query)
            
            # reduce embeddings dimensions
            scaled_embeddings = dimension_reduction(embeddings)
             
            # get model prediction and confidence score
            y_pred, confidence_score = perform_ml_operations(config, db, scaled_embeddings)
             
            # combine message metadata with model result
            data_metadata = generate_metadata(data[config.data.metadata_column], y_pred, confidence_score, config.models.confidence_score_threshold)
            
            # write to vector database
            vectorstore.write_to_vectorstore(zip(data[config.data.target_column].to_list(), embeddings), embedding_model, data_metadata) 
            
            update_metadata(config)
            
    except Exception as e:
        raise Exception(e)
    finally:            
        db.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument("--mlflow_uri", type=str, default='file:./mlruns', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument("--model_id", type=str, default=None, help='specify trained model, else use new model') 
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)