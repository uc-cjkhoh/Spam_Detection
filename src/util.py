import os
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine

from .decorators import timer, error_log
from loader.config_loader import cfg 
from loader.logger_loader import logging
            

@error_log
@timer
def create_required_folder_file():
    """
    Create necessary directorlies and files
    """
    
    # create directories 
    os.makedirs(cfg.models.save_model_to.folder, exist_ok=True)
    os.makedirs(cfg.hnsw.folder, exist_ok=True) 
    
    # create files
    if not os.path.isfile(cfg.module_log.process_log_path.files.label_record_file):
        pd.DataFrame(columns=cfg.active_learning.column_name).to_excel(
            cfg.module_log.process_log_path.files.label_record_file, index=False
        )
    if not os.path.isfile(cfg.module_log.process_log_path.files.unlabel_record_file):
        pd.DataFrame(columns=cfg.active_learning.column_name).to_excel(
            cfg.module_log.process_log_path.files.unlabel_record_file, index=False
        ) 
        
    
@error_log
@timer
def update_metadata(all_metadata: pd.DataFrame=None):
    """
    Update training process by checking if a subdata has been labelled.

    Args:
        metadata (pd.DataFrame): all subdata grouping metadata

    Returns:
        pd.DataFrame: pandas dataframe
    """
    
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
 

@error_log
@timer
def first_time_label():
    return len(pd.read_excel(cfg.module_log.process_log_path.files.label_record_file)) == 0


@error_log
@timer
def is_finish_labelling(metadata):
    finished_metadatas = pd.read_excel(cfg.module_log.process_log_path.files.label_record_file)
    return len(finished_metadatas) != 0 and np.any(np.all(finished_metadatas.to_numpy() == metadata, axis=1))


@error_log
@timer
def save_data(data: pd.DataFrame, vector: np.ndarray):
    # upload vector to mysql
    engine = create_engine(
        f'mysql+pymysql://{cfg.server.user}:{cfg.server.password}@{cfg.server.host}:{cfg.server.port}/sms_spam_cd'
    )
    
    # combine vector into dataframe
    data['embedding'] = [json.dumps(v.tolist()) for v in vector]
    
    columns_to_ingest = [
        'message_id',
        'embedding',
        'spam_label',
        'confidence_score',
        'cluster_label'
    ]
    
    data[columns_to_ingest].to_sql(
        name='ml_spam_result',
        con=engine,
        schema='sms_spam_cd',
        if_exists='append',
        index=False
    )
    
    engine.dispose() 