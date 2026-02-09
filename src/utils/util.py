import os    
import numpy as np
import pandas as pd  

from collections import Counter
from sklearn.cluster import HDBSCAN 
from imblearn.over_sampling import SMOTE   
     
from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 


@task(name='Create require files and directories', cache_policy=NO_CACHE)
def create_require_files_and_directories(config): 
    """Create require files and directories, if any
    """
    
    logger = get_run_logger()
    
    try:
        logger.info('Create vectorstore directory')
        os.makedirs(config.vectorstore.directory, exist_ok=True) 
     
    except PermissionError as e:
        logger.error(f"Permission error happened when creating files and directories: {e}", exc_info=True)
        raise
    
  
@task(name='Stratified Sampling', cache_policy=NO_CACHE)
def stratified_sampling(config, db) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Stratified Sampling

    Args:
        config (PyYAML): local configuration
        db (Database): database object

    Returns:
        tuple[pd.Series, pd.Series, pd.DataFrame]: id, datetime, payload from MySQL 
    """
    
    data = db.get_records(config.stratified_sampling)
    return data['id'], data['current_datetime'], data['payload']
  

@task(name='Oversampling', cache_policy=NO_CACHE)
def oversampling(x: np.ndarray, y: np.ndarray, k_neighbors=3) -> tuple[np.ndarray, np.ndarray]:
    """Perform oversampling

    Args:
        x (np.ndarray): independent features
        y (np.ndarray): dependent feature

    Returns:
        tuple[np.ndarray, np.ndarray]: oversampled x and y
    """
    
    counts = Counter(y)
    min_class = min(counts.values())
    
    if min_class < 3:
        return x, y
    
    smote = SMOTE(k_neighbors=min(k_neighbors, min_class - 1), random_state=42)
    resampled_x, resampled_y = smote.fit_resample(x, y)    
    return resampled_x, resampled_y

 
@task(name='Remove duplicate patterns', cache_policy=NO_CACHE)
def get_unique_pattern_ids(embeddings: np.ndarray) -> list[int]: 
    hdb = HDBSCAN()
    cluster_id = hdb.fit_predict(embeddings)
    ids = np.arange(0, embeddings.shape[0], 1)
    
    retent_df = pd.DataFrame({'ids': ids, 'cluster_id': cluster_id})
    
    non_outlier_df = retent_df[retent_df.cluster_id != -1]
    outlier_df = retent_df[retent_df.cluster_id == -1]
    
    unique_non_outlier_ids = non_outlier_df.groupby('cluster_id').head(5)['ids'].to_numpy(dtype=int)
    unique_outlier_ids = outlier_df['ids'].to_numpy(dtype=int)
    
    retent_ids = np.hstack((unique_non_outlier_ids, unique_outlier_ids))
    
    return retent_ids