import pandas as pd 
import numpy as np
import joblib
import os 

from tqdm import tqdm 
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.linear_model import SGDClassifier
from typing import Tuple

from loader.config_loader import cfg
from loader.logger_loader import logging
from .decorators import timer, error_log
from .util import has_availabel_model 
  

_model = SentenceTransformer(cfg.models.text_embedding.model_name, trust_remote_code=True)
_pipe = pipeline('text-classification', model=cfg.models.spam_detection.model_name)
_predictive_model = SGDClassifier(loss='log_loss')


@error_log
@timer
def text_embedding(messages: pd.Series) -> np.ndarray:
    return _model.encode(
        messages,  
        batch_size=cfg.models.text_embedding.batch_size, 
        show_progress_bar=True
    )
    
    
@error_log
@timer
def initial_labeling(data: pd.Series) -> pd.DataFrame: 
    def text_pipe(texts: pd.DataFrame, batch_size=16):
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            results.extend(_pipe(batch))
        return results
      
    prediction = text_pipe(data.to_list(), cfg.models.spam_detection.batch_size) 
    label = LabelEncoder()
    label = label.fit_transform([p['label'] for p in prediction])
    score = [p['score'] for p in prediction]
    
    return pd.DataFrame({
        cfg.data.target_column: data,
        cfg.data.target_column + '_label': label,
        cfg.data.target_column + '_score': score
    })
    
    
@error_log
@timer
def train_model(x_train: np.ndarray=None, y_train: np.ndarray=None, x_test: np.ndarray=None) -> Tuple[SGDClassifier, np.ndarray, np.ndarray]:
    if not has_availabel_model(type(_predictive_model).__name__):
        model = _predictive_model.fit(x_train, y_train)
    
        y_test = model.predict(x_test)
        confidence_score = model.predict_proba(x_test)
        return model, y_test, confidence_score.max(axis=1)
    else:
        version = datetime.now().strftime('%Y%m%d') 
        to_folder = cfg.models.save_model_to.folder
        filename = f'{type(_predictive_model).__name__}-{version}.joblib' 
        filepath = os.path.join(to_folder, filename)
        
        model = joblib.load(filepath)
        
        y_test = model.predict(x_test)
        confidence_score = model.predict_proba(x_test)
        return model, y_test, confidence_score.max(axis=1)
     

@error_log
@timer
def save_model(model):
    version = datetime.now().strftime('%Y%m%d') 
    to_folder = cfg.models.save_model_to.folder
    filename = f'{type(model).__name__}-{version}.joblib' 
    filepath = os.path.join(to_folder, filename)
    
    joblib.dump(model, filepath)
    logging.info(f'Save model to: {filepath}')
    
    
@error_log
@timer
def update_model(model, x, y):
    version = datetime.now().strftime('%Y%m%d') 
    to_folder = cfg.models.save_model_to.folder
    filename = f'{type(model).__name__}-{version}.joblib' 
    filepath = os.path.join(to_folder, filename)
    
    model.partial_fit(x, y)
    
    joblib.dump(model, filepath)
    logging.info(f'Model {filename} was updated')
     