import numpy as np
import pandas as pd
import joblib
import shutil
import os  

from datetime import datetime
   
from loader.config_loader import cfg
from loader.logger_loader import logging
from .decorators import timer, error_log  
     
  
@error_log
@timer
def save_model(model, filename): 
    to_folder = cfg.models.save_model_to.folder 
    filepath = os.path.join(to_folder, filename)
    
    joblib.dump(model, filepath)
    logging.info(f'Saved {filename} to {filepath}')


@error_log
@timer
def fit_latest_data(model, data):
    x = data[cfg.data.target_column]
    y = data['spam_label']
    model.partial_fit(x, y, classes=np.unique(y))

    
@error_log
@timer
def update_model(model, data): 
    """
    Update model with new data after saving old model
    """  
    try:
        save_model(model, f'{type(model).__name__}-{datetime.now().strftime("%Y%m%d%H%M")}.joblib')
        fit_latest_data(model, data)
        save_model(model, f'{type(model).__name__}.joblib')
        
    except Exception as e:
        logging.error(f"Failed to update model: {str(e)}")
        raise RuntimeError(f"Failed due to {e}")

