# === Standard Imports ===
import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
import json
import ast

from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sqlalchemy.engine import create_engine

from src.preprocess import text_normalize
from src.util import update_metadata, create_required_folder_file, first_time_label, is_finish_labelling, save_data
from src.decorators import timer, error_log 
from src.llm import initial_labeling, text_embedding
from src.model import update_model, save_model
# from src.clustering import direct_clustering, HNSW, recluster_from_database

from loader.data_loader import get_connector
from loader.config_loader import cfg
from loader.logger_loader import logging
    
        
@error_log
@timer
def get_metadata():
    """
    Return latest unlabel metadata(s)

    Args:
        cursor (mysql.connector): connector to execute query

    Returns:
        pd.DataFrame: latest subdata grouping metadata
    """ 
    
    con = get_connector()
    cur = con.cursor()
    
    cur.execute(cfg.active_learning.subdata_metadata_query)
    metadata = pd.DataFrame(cur.fetchall(), columns=cfg.active_learning.column_name) 
    
    finished_metadata = pd.read_excel(f'{cfg.module_log.process_log_path.files.label_record_file}')
    
    if not finished_metadata.empty:
        metadata = metadata.merge(finished_metadata, how='left', indicator=True)
        metadata = metadata[metadata._merge == 'left_only']
        metadata = metadata.drop('_merge', axis=1)
    
    return metadata

    
@error_log
@timer
def get_subdata(metadata):
    con = get_connector()
    cur = con.cursor()
    
    subdata_query = cfg.data.query.format(*metadata)
    
    logging.info(f'Loading subdata from: {("{}/" * len(cfg.active_learning.column_name)).strip("/").format(*metadata)}')
    cur.execute(subdata_query)
    subdata = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
    
    return subdata

  
@error_log
@timer
def load_model(): 
    model_name = f'{type(SGDClassifier()).__name__}.joblib'
    model_folder = cfg.models.save_model_to.folder
    return joblib.load(os.path.join(model_folder, model_name)) 
    
    
@error_log
@timer
def spam_detection(model, vector):  
    spam_label = model.predict(vector)
    confidence_score = model.predict_proba(vector) 
    return spam_label, confidence_score.max(axis=1)


@error_log
@timer
def initialize_model(model, imported_data): 
    embeddings = []
    labels = []
    for row in tqdm(imported_data):
        embedding = row[0].decode('utf-8')
        label = row[1]
        
        embeddings.append(ast.literal_eval(embedding))
        labels.append(label)
        
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    model.fit(embeddings, labels)
    
    filename = f'{type(model).__name__}.joblib'
    filepath = cfg.models.save_model_to.folder
    
    destination = os.path.join(filepath, filename)
    joblib.dump(model, destination)
  

@error_log
@timer
def initialize_all_components(first_subdata: pd.DataFrame):
    curr_meta_list = get_metadata()
    update_metadata(curr_meta_list) 
    
    first_subdata = get_subdata(curr_meta_list.to_numpy()[0]) 
    message = first_subdata[cfg.data.target_column]
    vector = np.array(first_subdata[cfg.data.vector_column].tolist())
    
    spam_label, confidence_score = initial_labeling(message)     
    
    data = pd.DataFrame({
        'id': first_subdata['id'], 
        'spam_label': spam_label,
        'confidence_score': confidence_score,
        'cluster_label': None
    })
     
    # save_data(data, vector) 
    update_metadata()

  
@error_log
@timer
def start_active_learning():
    try:   
        con = get_connector()
        cur = con.cursor()
        
        curr_meta_list = get_metadata()
        update_metadata(curr_meta_list)  
        
        for metadata in tqdm(curr_meta_list.to_numpy()):
            # skip finished group
            if is_finish_labelling(metadata):
                continue
            
            # if model not initialized
            if len(os.listdir(cfg.models.save_model_to.folder)) == 0:
                model = SGDClassifier(loss='log_loss')
                cur.execute(cfg.initialize_model.query) 
                initialize_model(model, cur.fetchall())
            
            subdata = get_subdata(metadata)
            
            subdata = text_normalize(subdata.copy())
            
            vector = text_embedding(subdata[cfg.data.target_column])
            
            model = joblib.load('models/SGDClassifier.joblib')
            
            is_spam, confidence_score = spam_detection(model, vector)
            
            update_model(model, vector, is_spam, confidence_score)
            
            save_data(subdata, vector, is_spam, confidence_score)
            
            update_metadata()
            
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)   
    finally:  
        logging.info(f'===== Module Ended At: {datetime.now()} =====')
        

if __name__ == '__main__':
    create_required_folder_file()   
    
    if first_time_label():
        initialize_all_components()
    else:
        start_active_learning()