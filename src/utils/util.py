import os  
import pandas as pd  

from langchain_huggingface import HuggingFaceEmbeddings 

from src.data_loader.connection import Database
from src.vector_database.vectorstore import VectorStore 
from src.ml.model_training import SGD, XGBoost
from src.config_folder.config_loader import get_config 
 
def setup_core_components(args): 
    config = get_config() 
    
    create_required_folder_file()
    
    database = Database(
        host="10.168.51.196",
        port=3306,
        user='unified',
        password='unified'
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={
            'batch_size': 8,
            'normalize_embeddings': True
        },
        show_progress=True
    )
    
    vectorstore = VectorStore(
        directory='./data/vector', 
        filename='sms_embeddings',
        embedding=embedding_model
    )
    
    teacher = SGD(experiment_name=args.experiment, model_name='Teacher') 
    student = SGD(experiment_name=args.experiment, model_name='Student')
      
    return config, database, embedding_model, vectorstore, teacher, student
 
def create_required_folder_file():  
    # create directories 
    # os.makedirs('logs/progress', exist_ok=True)
    os.makedirs('./data/vector', exist_ok=True)
    os.makedirs('./logs/evaluation', exist_ok=True)
    
    # # create files
    # if not os.path.isfile('logs/progress/label_metadata.xlsx'):
    #     pd.DataFrame(columns=['year', 'month', 'day', 'hour']).to_excel(
    #         'logs/progress/label_metadata.xlsx', index=False
    #     )
        
    # if not os.path.isfile('logs/progress/unlabel_metadata.xlsx'):
    #     pd.DataFrame(columns=['year', 'month', 'day', 'hour']).to_excel(
    #         'logs/progress/unlabel_metadata.xlsx', index=False
    #     ) 
         
    if not os.path.isfile('logs/evaluation/evaluation.xlsx'):
        pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Loss']).to_excel(
            'logs/evaluation/evaluation.xlsx', index=False
        ) 
 
def finish_labelling(filepath: str, label_column: str):
    if os.path.exists(filepath):
        data = pd.read_excel(filepath)
        if label_column in data.columns:
            if data[label_column].notna().all(): 
                return True 
    return False

def save_evaluation(new_data):
    filepath = "./logs/evaluation/evaluation.xlsx"
    file = pd.read_excel(filepath)
    file = pd.concat([file, new_data], ignore_index=True)
    file.to_excel(filepath, index=False)