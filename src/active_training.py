import pandas as pd 
import numpy as np
import os

from .decorators import error_log, timer 
from .model import text_embedding, train_model, save_model, update_model, _predictive_model
from .util import has_availabel_model

from loader.config_loader import cfg
from loader.logger_loader import logging


@error_log
@timer
def start_active_training(label_data: pd.DataFrame, unlabel_data: pd.DataFrame, to_be_fit_data: pd.DataFrame, threshold: float):
    x_test = text_embedding(unlabel_data[cfg.data.target_column])
     
    y_test = confidence_score = None
    if not has_availabel_model(type(_predictive_model).__name__):
        logging.info('No existing model, train a new one ...')
        x_train = text_embedding(label_data[cfg.data.target_column])
        y_train = label_data[cfg.data.target_column + '_label'].to_numpy() 
        
        model, y_test, confidence_score = train_model(x_train, y_train, x_test) 
        save_model(model)   
    else:
        logging.info('Have a existing model, fine-tuning ...')
        model, y_test, confidence_score = train_model(x_test=x_test)
    
    result_data = pd.DataFrame({
        cfg.data.target_column: unlabel_data[cfg.data.target_column],
        cfg.data.target_column + '_label': y_test,
        cfg.data.target_column + '_score': confidence_score
    })
    
    new_label_idx = np.where(confidence_score >= threshold)
    new_unlabel_idx = np.where(confidence_score < threshold)
        
    new_label_data = result_data.iloc[new_label_idx[0]]
    new_unlabel_data = result_data.iloc[new_unlabel_idx[0]]
    
    update_model(model, x_test[new_label_idx], y_test[new_label_idx])
    update_data_files(new_label_data, new_unlabel_data) 


@error_log 
@timer
def update_data_files(new_label: pd.DataFrame, new_unlabel: pd.DataFrame):
    label_filepath = cfg.active_learning.label_data_file
    unlabel_filepath = cfg.active_learning.unlabel_data_file
    
    old_label = pd.read_excel(label_filepath)
    old_unlabel = pd.read_excel(unlabel_filepath)
    
    updated_label_data = pd.concat([old_label, new_label])
    updated_unlabel_data = pd.concat([old_unlabel, new_unlabel])
    
    updated_label_data.to_excel(label_filepath, index=False)
    updated_unlabel_data.to_excel(unlabel_filepath, index=False)
    
    # remove all data in to_fit.xlsx
    pd.DataFrame(columns=[cfg.data.target_column, cfg.data.target_column + '_label', cfg.data.target_column + '_score']).to_excel(
        cfg.active_learning.to_be_fit_file, index=False
    )