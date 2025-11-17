import os 
import pandas as pd 
import numpy as np
 
from prefect import task 
from prefect.cache_policies import NO_CACHE  
from langchain_huggingface import HuggingFaceEmbeddings 
     
from data_loader.connection import Database
from src.vector_database.vectorstore import VectorStore


@task(cache_policy=NO_CACHE)  
def setup_environment(config: dict): 
    database = Database(
        host=config.server.host,
        port=config.server.port,
        user=config.server.user,
        password=config.server.password
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.models.text_embedding.model_name,
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 4
        },
        show_progress=True
    )
    
    vectorstore = VectorStore(
        directory=config.vectorstore.directory, 
        filename=config.vectorstore.filename
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
def generate_metadata(default_metadata: pd.DataFrame, ml_label, ml_confidence_score, threshold): 
    num_to_label = int(len(default_metadata) * 0.1)
    k_least_conf_idx = np.argpartition(ml_confidence_score, num_to_label)[:num_to_label] 
    label_status = np.array(['unlabeled'] * len(default_metadata), dtype=object)
    label_status[ml_confidence_score > threshold] = 'pseudo'
    label_status[k_least_conf_idx] = 'human'
         
    other_metadata = pd.DataFrame({
        'label': ml_label,
        'confidence_score': ml_confidence_score,
        'label_status': label_status
    }) 
    
    return pd.concat([default_metadata, other_metadata], axis=1).to_dict(orient='records')