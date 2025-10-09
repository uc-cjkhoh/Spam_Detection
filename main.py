# === Standard Imports ===
import pandas as pd
import numpy as np
import joblib
import os
import faiss
import json
import ast

from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sqlalchemy.engine import create_engine

# === Project Imports ===
from src.preprocess import text_normalize
from src.util import update_metadata, create_required_folder_file, first_time_label, is_finish_labelling, save_data
from src.decorators import timer, error_log 
from src.llm import initial_labeling, text_embedding
from src.model import update_model, save_model
from src.clustering import direct_clustering, HNSW
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
def initialize_first_batch_data(first_subdata: pd.DataFrame):
    message = first_subdata[cfg.data.target_column]
    vector = np.array(first_subdata[cfg.data.vector_column].tolist())
    
    spam_label, confidence_score = initial_labeling(message)     
    spam_message_idx = np.where(spam_label == 1)[0]
    spam_vector = vector[spam_message_idx]
    spam_cluster = direct_clustering(spam_vector)
    
    data = pd.DataFrame({
        'message_id': first_subdata['id'],
        'embedding': [json.dumps(v.tolist()) for v in vector], 
        'spam_label': spam_label,
        'confidence_score': confidence_score,
        'cluster_label': None
    })
     
    data.loc[spam_message_idx, 'cluster_label'] = spam_cluster
    
    engine = create_engine(
        f'mysql+pymysql://{cfg.server.user}:{cfg.server.password}@{cfg.server.host}:{cfg.server.port}/sms_spam_cd'
    )
    
    data.to_sql(
        name='ml_spam_result',
        con=engine,
        schema='sms_spam_cd',
        if_exists='append',
        index=False
    )
    
    engine.dispose()


@error_log
@timer
def initialize_model(model, x, y):
    model.fit(x, y)
    
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
def spam_detection(model, vector):  
    spam_label = model.predict(vector)
    confidence_score = model.predict_proba(vector)
    return spam_label, confidence_score.max(axis=1)

  
@error_log
@timer
def main(db_cursor):
    create_required_folder_file()   
    hnsw = HNSW(db_cursor)
    
    for metadata in tqdm(get_metadata(db_cursor).to_numpy()):
        if is_finish_labelling(metadata):
            logging.info(f'Skipping labelled metadata {("{}/" * len(cfg.active_learning.column_name)).strip("/").format(*metadata)}')
            continue
        
        subdata = get_subdata(db_cursor, metadata)[:100]
        subdata = text_normalize(subdata.copy())  
        
        vector = text_embedding(subdata[cfg.data.target_column])
        subdata[cfg.data.vector_column] = vector.tolist()
         
        if first_time_label():
            initialize_first_batch_data(subdata)  
            
            try:
                hnsw.initial(vector)
                faiss.write_index(hnsw, os.path.join(cfg.hnsw.folder, cfg.hnsw.filename))
            except Exception as e:
                logging.error(e, exc_info=True)
            
            update_metadata()
            logging.info('\nInitialization Successed ...')
            return 0
        else: 
            if len(os.listdir(cfg.models.save_model_to.folder)) == 0:
                logging.info('Initializing model ...')
                db_cursor.execute(cfg.query_selection.vector_label)
                
                imported_data = db_cursor.fetchall()
                
                embeddings = []
                labels = []
                for row in tqdm(imported_data):
                    embedding = row[0].decode('utf-8')
                    label = row[1]
                    
                    embeddings.append(ast.literal_eval(embedding))
                    labels.append(label)
                    
                embeddings = np.array(embeddings)
                labels = np.array(labels)
                
                initialize_model(
                    SGDClassifier(loss='modified_huber'),
                    x=embeddings,
                    y=labels
                )
            
            # load latest model
            logging.info('loading model ...')
            spam_detector = load_model()
            
            # detect spam 
            logging.info('start spam detection ...')
            spam_label, confidence_score = spam_detection(spam_detector, vector) 
            subdata['spam_label'] = spam_label
            subdata['confidence_score'] = confidence_score 
            
            try:
                # load hnsw
                logging.info('building hnsw ...')
                faiss.load_index(os.path.join(cfg.hnsw.folder, cfg.hnsw.filename))
                
                # clustering
                spam_message_idx = np.where(subdata['spam_label'] == 1)[0]
                spam_vector = vector[spam_message_idx]
                subdata.loc[spam_message_idx, 'cluster_label'] = hnsw.cluster_and_save(spam_vector)
            except Exception as e:
                logging.error(e, exc_info=True)
                
            # update model with high confidence labelled data
            high_confidence_data = subdata[subdata['confidence_score'] > cfg.models.spam_detection.labelling_confidence_threshold]
            x = np.array(high_confidence_data[cfg.data.vector_column].tolist())
            y = high_confidence_data['spam_label']
            
            update_model(spam_detector, x, y)
            save_model(spam_detector, filename=f'{type(spam_detector).__name__}.joblib')
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
        cur.close()
        connector.close()
        logging.info(f'===== Module Ended At: {datetime.now()} =====')