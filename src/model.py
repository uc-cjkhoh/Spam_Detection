import numpy as np
import pandas as pd
import joblib
import shutil
import os  

from datetime import datetime
   
from loader.config_loader import cfg
from loader.logger_loader import logging
from .decorators import timer, error_log
from .util import has_availabel_model, update_data_files
from .llm import text_embedding
     
  
class Custom_Models:
    def __init__(self, model):
        self.model = model
         
    @error_log
    @timer
    def train_model(self, model, x_train: np.ndarray=None, y_train: np.ndarray=None, x_test: np.ndarray=None): 
        if not has_availabel_model(type(model).__name__):
            model.fit(x_train, y_train)
        
            y_test = model.predict(x_test)
            confidence_score = model.predict_proba(x_test)
            return y_test, confidence_score.max(axis=1)
        else: 
            to_folder = cfg.models.save_model_to.folder
            filename = f'{type(model).__name__}.joblib' 
            filepath = os.path.join(to_folder, filename)
            
            model = joblib.load(filepath)
            
            y_test = model.predict(x_test)
            confidence_score = model.predict_proba(x_test)
            return y_test, confidence_score.max(axis=1)
    
    @error_log
    @timer
    def store_old_model(self, to_folder=cfg.models.save_model_to.folder): 
        try:
            # Get current model path
            filename = f'{type(self.model).__name__}.joblib'
            model_path = os.path.join(to_folder, filename)
            
            if not os.path.exists(model_path):
                logging.warning(f"No existing model found at {model_path}")
                return False
                
            # Create backup filename with timestamp
            creation_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            new_filename = f'{creation_time}-{type(self.model).__name__}.joblib'
            save_path = os.path.join(to_folder, new_filename)
            
            # Save current model state as backup
            joblib.dump(self.model, save_path)
            logging.info(f'Model saved to: {save_path}')
            return True
            
        except Exception as e:
            logging.error(f"Failed to store model backup: {str(e)}")
            raise

    @error_log
    @timer
    def save_model(self): 
        to_folder = cfg.models.save_model_to.folder
        filename = f'{type(self.model).__name__}.joblib' 
        filepath = os.path.join(to_folder, filename)
        
        joblib.dump(self.model, filepath)
        logging.info(f'Saved to: {filepath}')
        
    @error_log
    @timer
    def update_model(self, x, y): 
        """Update model with new data after creating backup"""
        try:
            to_folder = cfg.models.save_model_to.folder
            filename = f'{type(self.model).__name__}.joblib' 
            filepath = os.path.join(to_folder, filename)
            
            # Store current model state before updating
            backup_created = self.store_old_model(to_folder)
            if backup_created:
                logging.info("Created backup of current model state")
            
            # Update model with new data
            self.model.partial_fit(x, y, classes=np.unique(y))
            
            # Save updated model
            joblib.dump(self.model, filepath)
            logging.info(f'Model {filename} updated with {len(x)} new samples')
            
        except Exception as e:
            logging.error(f"Failed to update model: {str(e)}")
            raise

    @error_log
    @timer
    def check_any_data_to_fit(self):
        to_be_fit_folder = cfg.active_learning.to_be_fit_folder 
    
        for _file in os.listdir(to_be_fit_folder):
            filepath = os.path.join(to_be_fit_folder, _file)
            to_be_fit_data = pd.read_excel(filepath)
              
            logging.info(f'Fitting {_file} into {type(self.model).__name__}')
            self.update_model(
                model=self.model, 
                x=text_embedding(to_be_fit_data[cfg.data.target_column]), 
                y=to_be_fit_data[cfg.data.target_column + '_label']
            )
            
            shutil.move(filepath, cfg.active_learning.done_fit_folder)
            
            logging.info(f'Done fitting {_file} into {type(self.model).__name__}')
        
    @error_log
    @timer
    def start_active_training(self, label_data: pd.DataFrame, unlabel_data: pd.DataFrame, threshold: float):
        self.check_any_data_to_fit()
        
        if not has_availabel_model(type(self.model).__name__): 
            x_train = text_embedding(label_data[cfg.data.target_column])
            y_train = label_data[cfg.data.target_column + '_label'].to_numpy() 
            
        x_test = text_embedding(unlabel_data[cfg.data.target_column])
        y_test = None
        confidence_score = None
               
        if not has_availabel_model(type(self.model).__name__):
            logging.info(f'No existing {type(self.model).__name__} model, training one ...')  
            y_test, confidence_score = self.train_model(self.model, x_train, y_train, x_test) 
            self.save_model()   
        else:
            logging.info(f'Found existing {type(self.model).__name__}, loading model ...')
            y_test, confidence_score = self.train_model(model=self.model, x_test=x_test)
        
        result_data = pd.DataFrame({
            cfg.data.target_column: unlabel_data[cfg.data.target_column],
            cfg.data.target_column + '_label': y_test,
            cfg.data.target_column + '_score': confidence_score
        })
        
        new_label_idx = np.where(confidence_score >= threshold)
        new_unlabel_idx = np.where(confidence_score < threshold)
            
        new_label_data = result_data.iloc[new_label_idx[0]]
        new_unlabel_data = result_data.iloc[new_unlabel_idx[0]]
        
        self.update_model(x_test[new_label_idx], y_test[new_label_idx])
        update_data_files(new_label_data, new_unlabel_data)
