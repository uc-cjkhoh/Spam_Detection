import os 
import pandas as pd 
 
from prefect import task 
from prefect.cache_policies import NO_CACHE  
     
     
@task(cache_policy=NO_CACHE)      
def create_required_folder_file(cfg):
    """
    Create necessary directorlies and files
    """
    
    # create directories 
    os.makedirs(cfg.progress_log.folder, exist_ok=True)
    os.makedirs(cfg.vectorstore.directory, exist_ok=True)
    
    # create files
    if not os.path.isfile(cfg.progress_log.files.finished):
        pd.DataFrame(columns=cfg.metadata.column_name).to_excel(
            cfg.progress_log.files.finished, index=False
        )
        
    if not os.path.isfile(cfg.progress_log.files.unfinished):
        pd.DataFrame(columns=cfg.metadata.column_name).to_excel(
            cfg.progress_log.files.unfinished, index=False
        ) 

 
def update_metadata(config: dict, all_metadata: pd.DataFrame=None):
    """
    Update training process by checking if a subdata has been labelled.

    Args:
        metadata (pd.DataFrame): all subdata grouping metadata

    Returns:
        pd.DataFrame: pandas dataframe
    """
    
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