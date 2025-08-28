import numpy as np
import pandas as pd
import joblib
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
    def store_old_model(self, model, to_folder):
        filename = f'{type(model).__name__}.joblib'
        model_path = os.path.join(to_folder, filename)
        creation_time = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d_%H_%M_%S")
    
        new_filename = f'{creation_time}-{type(model).__name__}.joblib'
        to_path = os.path.join(to_folder, new_filename)
        
        joblib.dump(model, to_path)
        logging.info(f'Old mode saved to: {to_path}')
    
    @error_log
    @timer
    def save_model(self, model): 
        to_folder = cfg.models.save_model_to.folder
        filename = f'{type(model).__name__}.joblib' 
        filepath = os.path.join(to_folder, filename)
        
        joblib.dump(model, filepath)
        logging.info(f'Saved to: {filepath}')
        
    @error_log
    @timer
    def update_model(self, model, x, y): 
        to_folder = cfg.models.save_model_to.folder
        filename = f'{type(model).__name__}.joblib' 
        
        self.store_old_model(model, to_folder)
        model.partial_fit(x, y, classes=np.unique(y))
        
        filepath = os.path.join(to_folder, filename)
        joblib.dump(model, filepath)
        logging.info(f'Model {filename} was updated')
    
    @error_log
    @timer
    def check_any_data_to_fit(self):
        to_be_fit_folder = cfg.active_learning.to_be_fit_folder
        to_be_fit_data = None 
    
        for _file in os.listdir(to_be_fit_folder):
            if to_be_fit_data is None:
                to_be_fit_data = pd.read_excel(os.path.join(to_be_fit_folder, _file))
            else:
                to_be_fit_data = pd.concat([to_be_fit_data, pd.read_excel(os.path.join(to_be_fit_folder, _file))])                        
                 
        if to_be_fit_data is not None:
            logging.info(f'Fitting data into {type(self.model).__name__}')
            self.update_model(
                model=self.model, 
                x=text_embedding(to_be_fit_data[cfg.data.target_column]), 
                y=to_be_fit_data[cfg.data.target_column + '_label']
            )
            # remove all data in to_fit.xlsx
            pd.DataFrame(columns=[cfg.data.target_column, cfg.data.target_column + '_label', cfg.data.target_column + '_score']).to_excel(
                cfg.active_learning.to_be_fit_file, index=False
            )
            logging.info(f'Done fitting data into {type(self.model).__name__}')
        else:
            logging.info("No data waiting to be fit.") 
        
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
            self.save_model(self.model)   
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
        
        self.update_model(self.model, x_test[new_label_idx], y_test[new_label_idx])
        update_data_files(new_label_data, new_unlabel_data) 
        