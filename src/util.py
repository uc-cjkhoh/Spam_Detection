import os
import pandas as pd

from . decorators import timer, error_log

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
def check_learning_process(metadata: pd.DataFrame):
    log_folder = './logs'
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        
    finished_subgroup_file = f'{log_folder}/finished_subdata.xlsx'
    
    # if finished.xlsx not exists or no data inside
    if not os.path.exists(finished_subgroup_file) or os.path.getsize(finished_subgroup_file) == 0:
        metadata.to_excel(f'{log_folder}/unfinish_subdata.xlsx', index=False)
    else:
        finished_subgroup = pd.read_excel(finished_subgroup_file)
        
        metadata = metadata.merge(finished_subgroup, how='left', indicator=True)
        metadata = metadata[metadata._merge == 'left_only']
        
        metadata.to_excel(f'{log_folder}/unfinish_subdata.xlsx')
    
    return metadata