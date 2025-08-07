import os
import pandas as pd

from pathlib import Path

from . decorators import timer, error_log
from . model import train_model
from loader.config_loader import cfg 

@error_log
@timer
def setup_directory_and_file():
    """
    Create necessary directorlies and files
    """
    
    # create directory
    os.makedirs(cfg.active_learning.label_data_folder, exist_ok=True)
    os.makedirs(cfg.active_learning.unlabel_data_folder, exist_ok=True) 
    
    # create file
    Path(f'{cfg.module_log.process_log_path.files.label_record_file}').touch(exist_ok=True)
    Path(f'{cfg.module_log.process_log_path.files.unlabel_record_file}').touch(exist_ok=True)
    
        
@error_log 
@timer
def is_folder_empty(folderpath: str) -> bool:
    """
    Check if folder is empty

    Args:
        folderpath (str): target folder's path

    Returns:
        bool: True if folder is empty else False
    """
    
    if not os.path.isdir(folderpath):
        print(f'Folder {folderpath} not exists, creating folder ...')
        return False
    
    return not os.listdir(folderpath)


@error_log
@timer
def update_unfinish_metadata(metadata: pd.DataFrame):
    """
    Update training process by checking if a subdata has been labelled.

    Args:
        metadata (pd.DataFrame): all subdata grouping metadata

    Returns:
        pd.DataFrame: pandas dataframe
    """
    
    finished_subgroup_file = f'{cfg.module_log.process_log_path.files.label_record_file}'
    
    if os.path.getsize(finished_subgroup_file) == 0:
        metadata.to_excel(f'{cfg.module_log.process_log_path.files.unlabel_record_file}', index=False)
    else:
        finished_subgroup = pd.read_excel(finished_subgroup_file)
        
        metadata = metadata.merge(finished_subgroup, how='left', indicator=True)
        metadata = metadata[metadata._merge == 'left_only']
        metadata = metadata.drop('_merge', axis=1)
        metadata.to_excel(f'{cfg.module_log.process_log_path.files.unlabel_record_file}', index=False)
      
      
@error_log
@timer
def initial_first_label_set(data: pd.DataFrame, subdata_metadata: pd.DataFrame, metadata: list):
    label_data = train_model(data[cfg.data.target_column]) 
    label_data.to_excel(
        f'{cfg.active_learning.label_data_folder.strip("/")}/{cfg.active_learning.label_data_filename}', 
        index=False
    )
    
    current_metadata = pd.DataFrame([metadata], columns=cfg.active_learning.column_name)
    finished_metadata_file = f'{cfg.module_log.process_log_path.files.label_record_file}'
    
    if os.path.getsize(finished_metadata_file) == 0:
        current_metadata.to_excel(finished_metadata_file, index=False)
    else:
        with pd.ExcelWriter(finished_metadata_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            current_metadata.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

    update_unfinish_metadata(subdata_metadata.drop(index=0))
    