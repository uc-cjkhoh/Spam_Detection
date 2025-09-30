# === Standard Imports ===
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import sys
import os

from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# === Project Imports ===
from src.preprocess import text_normalize
from src.util import update_metadata, create_required_folder_file, first_time_label, is_finish_labelling, save_data
from src.decorators import timer, error_log 
from src.llm import initial_labeling, text_embedding
from src.model import update_model, save_model
from src.clustering import get_clustering
# from src.eda import basic_eda

from loader.data_loader import Database
from loader.config_loader import cfg
from loader.logger_loader import logging

  

@error_log
@timer
def get_metadata(cur):
    all_metadata = database.get_metadata(cur)
    update_metadata(all_metadata)
    
    return all_metadata
  
  
@error_log
@timer
def get_subdata(cur, metadata):
    subdata_query = cfg.data.query.format(*metadata)
    
    logging.info(f'Loading subdata from: {("{}/" * len(cfg.active_learning.column_name)).strip("/").format(*metadata)}')
    cur.execute(subdata_query)
    subdata = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
    
    return subdata


@error_log
@timer
def initialize_first_batch_data(message):
    file_type = ['xlsx', 'npy']
    
    vector = text_embedding(message)
    
    initial_batch = initial_labeling(message)    
     
    spam_message_idx = np.where(initial_batch['spam_label'] == 1)[0]
    spam_vector = vector[spam_message_idx]
    spam_cluster = get_clustering(spam_vector)
      
    initial_batch.loc[spam_message_idx, 'cluster_label'] = spam_cluster
    
    data_to_saved = [initial_batch, vector] 
    data_folders = [cfg.active_learning.raw_message_folder, cfg.active_learning.message_vector_folder]
    
    for i, value in enumerate(data_to_saved):    
        filename = f'{len(os.listdir(data_folders[i]))}.{file_type[i]}' 
        destination = os.path.join(data_folders[i], filename)

        if file_type[i] == 'xlsx':
            value.to_excel(destination, index=False)
        else:
            np.save(destination, value) 


@error_log
@timer
def initialize_model(model):
    data_folders = [cfg.active_learning.raw_message_folder, cfg.active_learning.message_vector_folder]
    
    message_filepath = os.path.join(data_folders[0], '0.xlsx')
    spam_label = pd.read_excel(message_filepath)['spam_label']
    
    vector_filepath = os.path.join(data_folders[1], '0.npy')
    vector = np.load(vector_filepath)
    
    model.fit(X=vector, y=spam_label)
    
    filename = f'{type(model).__name__}.joblib'
    filepath = cfg.models.save_model_to.folder
    
    destination = os.path.join(filepath, filename)
    joblib.dump(model, destination)
  
  
@error_log
@timer
def load_model(): 
    model_name = f'{type(SGDClassifier()).__name__}.joblib'
    model_folder = cfg.models.save_model_to.folder
    return joblib.load(os.path.join(model_folder, model_name)) 
    
    
@error_log
@timer
def spam_detection(model, message):  
    spam_label = model.predict(message)
    confidence_score = model.predict_proba()
    return spam_label, confidence_score.max(axis=1)

  
@error_log
@timer
def main(db_cursor):
    create_required_folder_file()   
    
    for metadata in tqdm(get_metadata(db_cursor).to_numpy()):
        if is_finish_labelling(metadata):
            logging.info(f'Skipping labelled metadata {("{}/" * len(cfg.active_learning.column_name)).strip("/").format(*metadata)}')
            continue
        
        subdata = get_subdata(db_cursor, metadata) 
        subdata = text_normalize(subdata.copy())  
         
        if first_time_label():
            initialize_first_batch_data(subdata[cfg.data.target_column])  
            
            update_metadata()
            logging.info('\nInitialization Successed ...')
            return 0
        else: 
            initialize_model(SGDClassifier(loss='modified_huber'))
            
            # load latest model
            latest_model = load_model()
            
            # start detecting spam and clusters
            spam_label, confidence_score = spam_detection(latest_model, subdata[cfg.data.target_column])
            
            # update model with high confidence result
            threshold=cfg.models.spam_detection.labelling_confidence_threshold
            high_confidence_message = subdata[subdata['confidence_score'] > threshold]      
            update_model(latest_model, high_confidence_message)   
            
            subdata['spam_label'] = spam_label
            subdata['confidence_score'] = confidence_score
            
            # cluster spam and non-spam message separately
            spam_message_idx = np.where(subdata['spam_label'] == 0)[0]
            spam_vector = vector[spam_message_idx]
            subdata.loc[spam_message_idx, 'cluster_label'] = get_clustering(spam_vector, cluster_limit=20)
             
            save_model(latest_model, filename=f'{type(latest_model).__name__}.joblib')
            save_data(subdata, vector)
            update_metadata()
        

if __name__ == '__main__':
    try:
        database = Database(
            host=cfg.server.host,
            port=cfg.server.port,
            user=cfg.server.user,
            password=cfg.server.password
        )

        connector = database.connect_db() 
        cur = connector.cursor()
        
        main(cur)
    except KeyboardInterrupt as e:
        logging.error(e, exc_info=True)
    finally:
        logging.info('===== End of Execution =====')