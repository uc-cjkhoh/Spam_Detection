import os  
import pandas as pd  

from langchain_huggingface import HuggingFaceEmbeddings 

from src.data_loader.connection import Database
from src.vector_database.vectorstore import VectorStore 
from src.ml.model_training import SGD 
from src.config_folder.config_loader import get_config 
 
def setup_core_components(): 
    config = get_config() 
    
    create_required_folder_file(config)
    
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
    
    model = SGD(config.mlflow_config.experiment_name)
     
    metadata = database.run_query(config.metadata.query, columns=config.metadata.column_name) 
     
    return config, database, metadata, embedding_model, vectorstore, model
 
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
 
def finish_labelling(filepath: str, label_column: str):
    if os.path.exists(filepath):
        data = pd.read_excel(filepath)
        if label_column in data.columns:
            if data[label_column].notna().all(): 
                return True 
    return False
