import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import json

from prefect import task
from prefect.cache_policies import NO_CACHE 
from sklearn.linear_model import SGDClassifier

#custom libraries
from src.data_loader.data_loader import Database  
from src.model.model_training import MLPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from src.vector_database.vectorstore import VectorStore

    
@task(cahce_policies=NO_CACHE)      
def create_required_folder_file(cfg):
    """
    Create necessary directorlies and files
    """
    
    # create directories 
    os.makedirs(cfg.module_log.process_log_path.folder, exist_ok=True)
    os.makedirs(cfg.vectorstore.directory, exist_ok=True)
    
    # create files
    if not os.path.isfile(cfg.module_log.process_log_path.files.label_record_file):
        pd.DataFrame(columns=cfg.metadata.column_name).to_excel(
            cfg.module_log.process_log_path.files.label_record_file, index=False
        )
        
    if not os.path.isfile(cfg.module_log.process_log_path.files.unlabel_record_file):
        pd.DataFrame(columns=cfg.metadata.column_name).to_excel(
            cfg.module_log.process_log_path.files.unlabel_record_file, index=False
        ) 
        
   
@task
def setup_environment(config: dict, args): 
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
    
    ml_model = identify_model_to_train(args)
    ml_pipeline = MLPipeline(ml_model)
    
    return database, embedding_model, vectorstore, ml_pipeline
  

@task
def identify_model_to_train(args):
    model = SGDClassifier(loss='log_loss', class_weight='balanced') 
    
    if args.model_path is not None:
        model = joblib.load(args.model_path)
    else: 
        runs = mlflow.search_runs(
            order_by=['attributes.start_time DESC'],
            experiment_names=[args.experiment],
            max_results=1
        )
        
        if len(runs) > 0:
            latest_run_id = runs[0].info.run_id
            model_uri = f"run:/{latest_run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            
    return model

 