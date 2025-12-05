import os 
import numpy as np
import pandas as pd 
 
from prefect import task 
from prefect.cache_policies import NO_CACHE  
from langchain_huggingface import HuggingFaceEmbeddings 
     
from src.data_loader.connection import Database
from src.vector_database.vectorstore import VectorStore


@task(cache_policy=NO_CACHE)  
def setup_core_instances(config: dict): 
    database = Database(
        host=config.server.host,
        port=config.server.port,
        user=config.server.user,
        password=config.server.password
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.models.text_embedding.model_name, 
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 4
        },
        show_progress=True
    )
    
    vectorstore = VectorStore(
        directory=config.vectorstore.directory, 
        filename=config.vectorstore.filename,
        embedding=embedding_model
    )
     
    return database, embedding_model, vectorstore   


@task(cache_policy=NO_CACHE)      
def create_required_folder_file(config: dict):  
    # create directories 
    os.makedirs(config.progress_log.folder, exist_ok=True)
    os.makedirs(config.vectorstore.directory, exist_ok=True)
    
    # create files
    if not os.path.isfile(config.progress_log.files.finished):
        pd.DataFrame(columns=config.metadata.column_name).to_excel(
            config.progress_log.files.finished, index=False
        )
        
    if not os.path.isfile(config.progress_log.files.unfinished):
        pd.DataFrame(columns=config.metadata.column_name).to_excel(
            config.progress_log.files.unfinished, index=False
        ) 

 
@task(cache_policy=NO_CACHE)
def update_metadata(config: dict, all_metadata: pd.DataFrame=None): 
    finished_metadata = config.progress_log.files.finished
    unfinished_metadata = config.progress_log.files.unfinished
    
    if all_metadata is not None:
        all_metadata.to_excel(unfinished_metadata, index=False)
    else:
        finished = pd.read_excel(finished_metadata)
        unfinished = pd.read_excel(unfinished_metadata)
        
        updated_finished = pd.concat([finished, unfinished.iloc[0].to_frame().T])
        updated_unfinished = unfinished.drop(index=0)
        
        updated_finished.to_excel(finished_metadata, index=False)
        updated_unfinished.to_excel(unfinished_metadata, index=False)
        

@task(cache_policy=NO_CACHE)
def initialize_metadata(default_metadata: pd.DataFrame): 
    other_metadata = pd.DataFrame({
        'label': [None] * len(default_metadata),
        'confidence_score': [None] * len(default_metadata),
        'label_status': [None] * len(default_metadata)
    }) 
    
    return pd.concat([default_metadata, other_metadata], axis=1).to_dict(orient='records')


@task(cache_policy=NO_CACHE)
def generate_metadata(default_metadata: pd.DataFrame, y_pred: np.ndarray, confidence_score: np.ndarray, threshold: float):
    pseudo_idx = np.where(confidence_score > threshold)[0]
    human_idx = np.argpartition(confidence_score, int(len(default_metadata) * 0.1))[:int(len(default_metadata) * 0.1)]
    
    label_status = np.array(['unlabeled'] * len(default_metadata), dtype=object)
    label_status[pseudo_idx] = 'high_confidence'
    label_status[human_idx] = 'least_confidence'
    
    metadata = pd.concat([
        default_metadata,
        pd.DataFrame({
            'label': y_pred,
            'confidence_score': confidence_score,
            'label_status': label_status
        })
    ], axis=1).to_dict(orient='records')
    
    return metadata