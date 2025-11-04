import os
import pandas as pd
import numpy as np
import json

from sqlalchemy import create_engine
from testing.config_loader.config_loader import get_config
           
           
def create_required_folder_file():
    """
    Create necessary directorlies and files
    """
    
    cfg = get_config()
    
    # create directories 
    os.makedirs(cfg.module_log.general_log_path.folder, exist_ok=True)
    os.makedirs(cfg.module_log.process_log_path.folder, exist_ok=True)
    os.makedirs(cfg.models.save_model_to.folder, exist_ok=True)
    os.makedirs(cfg.hnsw.folder, exist_ok=True) 
    os.makedirs(cfg.vectorstore.default_location, exist_ok=True)
    
    # create files
    if not os.path.isfile(cfg.module_log.process_log_path.files.label_record_file):
        pd.DataFrame(columns=cfg.metadata.column_name).to_excel(
            cfg.module_log.process_log_path.files.label_record_file, index=False
        )
        
    if not os.path.isfile(cfg.module_log.process_log_path.files.unlabel_record_file):
        pd.DataFrame(columns=cfg.metadata.column_name).to_excel(
            cfg.module_log.process_log_path.files.unlabel_record_file, index=False
        ) 
        
     
def update_metadata(all_metadata: pd.DataFrame=None):
    """
    Update training process by checking if a subdata has been labelled.

    Args:
        metadata (pd.DataFrame): all subdata grouping metadata

    Returns:
        pd.DataFrame: pandas dataframe
    """
    
    cfg = get_config()
    
    finished_metadata = cfg.module_log.process_log_path.files.label_record_file
    unfinished_metadata = cfg.module_log.process_log_path.files.unlabel_record_file
    
    if all_metadata is not None:
        all_metadata.to_excel(unfinished_metadata, index=False)
    else:
        finished = pd.read_excel(finished_metadata)
        unfinished = pd.read_excel(unfinished_metadata)
        
        updated_finished = pd.concat([finished, unfinished.iloc[0].to_frame().T])
        updated_unfinished = unfinished.drop(index=0)
        
        updated_finished.to_excel(finished_metadata, index=False)
        updated_unfinished.to_excel(unfinished_metadata, index=False)
 
 
def first_time_label():
    cfg = get_config()
    return len(pd.read_excel(cfg.module_log.process_log_path.files.label_record_file)) == 0


def check_exist_model():
    return False
    
 
def is_finish_labelling(metadata):
    cfg = get_config()
    finished_metadatas = pd.read_excel(cfg.module_log.process_log_path.files.label_record_file)
    return len(finished_metadatas) != 0 and np.any(np.all(finished_metadatas.to_numpy() == metadata, axis=1))

 
def save_data(data: pd.DataFrame, vector: np.ndarray, is_spam, confidence_score):
    cfg = get_config()
    
    # upload vector to mysql
    engine = create_engine(
        f'mysql+pymysql://{cfg.server.user}:{cfg.server.password}@{cfg.server.host}:{cfg.server.port}/sms_spam_cd'
    )
    
    # combine vector into dataframe
    data['embedding'] = [json.dumps(v.tolist()) for v in vector]
    data['spam_label'] = is_spam
    data['confidence_score'] = confidence_score
    
    columns_to_ingest = [
        'id',
        'embedding',
        'spam_label',
        'confidence_score' 
    ]
    
    data[columns_to_ingest].to_sql(
        name='ml_spam_result',
        con=engine,
        schema='sms_spam_cd',
        if_exists='append',
        index=False
    )
    
    engine.dispose() 