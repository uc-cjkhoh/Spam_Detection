import numpy as np
import pandas as pd
import joblib 
import sys
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
def update_model(model, vector, is_spam, confidence_score): 
    """Update model with new data after saving old model

    Args:
        model (_type_): _description_
        data (_type_): _description_
    """
    
    try:
        save_model(model, f'{type(model).__name__}-{datetime.now().strftime("%Y%m%d%H%M")}.joblib')
          
        high_confidence_idx = np.where(confidence_score >= 0.9)[0]
        selected_vector = vector[high_confidence_idx].astype(np.float64)
        selected_spam_label = is_spam[high_confidence_idx].astype(np.float64)
         
        model.partial_fit(selected_vector, selected_spam_label)
        
        save_model(model, f'{type(model).__name__}.joblib')
        
    except Exception as e:
        logging.error(f"Failed to update model: {str(e)}")
        raise RuntimeError(f"Failed due to {e}")

