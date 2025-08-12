import os
import pandas as pd
import numpy as np

from .decorators import timer, error_log
from loader.config_loader import cfg 


@error_log
@timer
def setup_directory_and_file():
    """
    Create necessary directorlies and files
    """
    
    # create directories
    os.makedirs(cfg.active_learning.label_data_folder, exist_ok=True)
    os.makedirs(cfg.active_learning.unlabel_data_folder, exist_ok=True)
    os.makedirs(cfg.active_learning.to_be_fit_folder, exist_ok=True) 
    os.makedirs(cfg.models.save_model_to.folder, exist_ok=True) 
    
    # create files
    if not os.path.isfile(cfg.module_log.process_log_path.files.label_record_file):
        pd.DataFrame(columns=cfg.active_learning.column_name).to_excel(
            cfg.module_log.process_log_path.files.label_record_file, index=False
        )
    if not os.path.isfile(cfg.module_log.process_log_path.files.unlabel_record_file):
        pd.DataFrame(columns=cfg.active_learning.column_name).to_excel(
            cfg.module_log.process_log_path.files.unlabel_record_file, index=False
        )
    if not os.path.isfile(cfg.active_learning.label_data_file):
        pd.DataFrame(columns=[cfg.data.target_column, cfg.data.target_column + '_label', cfg.data.target_column + '_score']).to_excel(
            cfg.active_learning.label_data_file, index=False
        )
    if not os.path.isfile(cfg.active_learning.unlabel_data_file):
        pd.DataFrame(columns=[cfg.data.target_column, cfg.data.target_column + '_label', cfg.data.target_column + '_score']).to_excel(
            cfg.active_learning.unlabel_data_file, index=False
        )
    if not os.path.isfile(cfg.active_learning.to_be_fit_file):
        pd.DataFrame(columns=[cfg.data.target_column, cfg.data.target_column + '_label', cfg.data.target_column + '_score']).to_excel(
            cfg.active_learning.to_be_fit_file, index=False
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
def has_availabel_model(model_class: str):
    models_folderpath = cfg.models.save_model_to.folder
    
    if len(os.listdir(models_folderpath)) != 0:
        return model_class in [x.split('.')[0] for x in os.listdir(models_folderpath)][0]
    
    return False