import os    
import numpy as np
import pandas as pd  

from collections import Counter
from sklearn.cluster import HDBSCAN 
from imblearn.over_sampling import SMOTE   
     
from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from data_validation.utils_validation.validate_util import CreateFolderFilesConfig, \
    SamplingConfig, OversamplingInput, GetUniquePatternInput
    

@task(name='Create require files and directories', cache_policy=NO_CACHE)
def create_require_files_and_directories(config: CreateFolderFilesConfig): 
    """Create require files and directories, if any
    
        Args:
            config (CreateFolderFilesConfig): refer to CreateFolderFilesConfig in validate_util.py
    """
    
    logger = get_run_logger()
    
    try:
        try:
            logger.info('Create vectorstore directory')
            os.makedirs(config.vectorstore.directory, exist_ok=True) 
            
        except PermissionError as e:
            logger.error(f"Permission error happened when creating files and directories: {e}", exc_info=True)
            raise
        
    except Exception as e:
        logger.error(f"Unexpected error in create_require_files_and_directories: {e}", exc_info=True)
        raise


@task(name='Stratified Sampling', cache_policy=NO_CACHE)
def stratified_sampling(sampling_config: SamplingConfig) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Stratified Sampling

    Args:
        sampling_config (SamplingConfig): refer to SamplingConfig in validate_util.py

    Returns:
        tuple[pd.Series, pd.Series, pd.DataFrame]: payload ids, payload datetime, payloads
    """
    
    logger = get_run_logger()
    
    try:
        data = sampling_config.db.get_records(sampling_config.config.stratified_sampling)
        return data['id'], data['current_datetime'], data['payload']
    
    except Exception as e:
        logger.error(f"Error in stratified_sampling: {e}", exc_info=True)
        raise


@task(name='Oversampling', cache_policy=NO_CACHE)
def oversampling(oversampling_input: OversamplingInput) -> tuple[np.ndarray, np.ndarray]:
    """Perform Oversampling

    Args:
        oversampling_input (OversamplingInput): refer to OversamplingInput in validate_util.py

    Returns:
        tuple[np.ndarray, np.ndarray]: resampled independent, resample dependent
    """
    
    logger = get_run_logger()
    
    try:
        counts = Counter(oversampling_input.y)
        min_class = min(counts.values())
        
        if min_class < 3:
            return oversampling_input.x, oversampling_input.y
        
        smote = SMOTE(k_neighbors=min(oversampling_input.k_neighbors, min_class - 1), random_state=42)
        resampled_x, resampled_y = smote.fit_resample(oversampling_input.x, oversampling_input.y)    
        return np.asarray(resampled_x), np.asarray(resampled_y)
    
    except Exception as e:
        logger.error(f"Error in oversampling: {e}", exc_info=True)
        raise


@task(name='Remove duplicate patterns', cache_policy=NO_CACHE)
def get_unique_pattern_ids(get_unique_pattern_input: GetUniquePatternInput) -> list[int]: 
    """Retrieve id of payloads contain unique sms pattern

    Args:
        get_unique_pattern_config: GetUniquePatternInput

    Returns:
        list[int]: unique sms pattern id
    """
    
    logger = get_run_logger()
    
    try:
        hdb = HDBSCAN()
        cluster_id = hdb.fit_predict(get_unique_pattern_input.embeddings)
        ids = np.arange(0, get_unique_pattern_input.embeddings.shape[0], 1)
        
        retent_df = pd.DataFrame({'ids': ids, 'cluster_id': cluster_id})
        
        non_outlier_df = retent_df[retent_df.cluster_id != -1]
        outlier_df = retent_df[retent_df.cluster_id == -1]
        
        unique_non_outlier_ids = non_outlier_df.groupby('cluster_id').head(5)['ids'].to_numpy(dtype=int)
        unique_outlier_ids = outlier_df['ids'].to_numpy(dtype=int)
        
        retent_ids = np.hstack((unique_non_outlier_ids, unique_outlier_ids))    
        return retent_ids
    
    except Exception as e:
        logger.error(f"Error in get_unique_pattern_ids: {e}", exc_info=True)
        raise
