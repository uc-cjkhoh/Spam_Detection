import os 
import ast
import numpy as np
import pandas as pd 

from tqdm import tqdm 
from langchain_huggingface import HuggingFaceEmbeddings 

from src.data_loader.connection import Database
from src.vector_database.vectorstore import VectorStore

 
def setup_core_instances(config: dict): 
    database = Database(
        host=config.server.host,
        port=config.server.port,
        user=config.server.user,
        password=config.server.password
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={
            'normalize_embeddings': True
        },
        show_progress=True
    )
    
    vectorstore = VectorStore(
        directory=config.vectorstore.directory, 
        filename=config.vectorstore.filename,
        embedding=embedding_model
    )
     
    return database, embedding_model, vectorstore   

   
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


def faiss_index_exists(folder_path: str, index_name: str):
    default_faiss_path = os.path.join(folder_path, index_name)
    default_faiss_extensions = ['.faiss', 'pkl']
    
    for ext in default_faiss_extensions:
        abs_path = default_faiss_path + ext
        if not os.path.exists(abs_path):
            return False 
    return True


def finish_labelling(filepath: str, label_column: str):
    if os.path.exists(filepath):
        data = pd.read_excel(filepath)
        if label_column in data.columns:
            if data[label_column].notna().all(): 
                return True
    
    return False