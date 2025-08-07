# === Standard Imports ===
import os
import sys
import pandas as pd
from tqdm import tqdm

# === Project Imports ===
from src.preprocess import text_normalize, feature_engineering
from src.model import train_model
from src.util import update_unfinish_metadata, is_folder_empty, setup_directory_and_file, initial_first_label_set
from src.active_training import start_active_training
from src.decorators import timer, error_log
# from src.eda import basic_eda

from loader.data_loader import Database
from loader.config_loader import cfg
from loader.logger_loader import logging
  
  
@error_log
@timer
def main():
    setup_directory_and_file()
    
    database = Database('mysql')
    connector = database.connect_db()
    cur = connector.cursor()
    
    subdata_metadata = database.get_metadata(cur)
    subdata_log_dt = ('{}/' * len(cfg.active_learning.column_name)).strip('/')
    
    # Starting core process 
    for metadata in tqdm(subdata_metadata.to_numpy()):
        query = cfg.data.query.format(*metadata)
        logging.info(f'Learning subdata of datetime: {subdata_log_dt.format(*metadata)}')
        
        cur.execute(query)
        unlabel_data = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
        
        # Try the best to normalize messages
        unlabel_data = text_normalize(unlabel_data.copy()) 
        
        # Check data statistic 
        if cfg.data.drop_null:
            unlabel_data = unlabel_data.dropna()
        if cfg.data.drop_duplicates:
            unlabel_data = unlabel_data.drop_duplicates()
          
        unlabel_data = unlabel_data.select_dtypes(include=['object'])
        
        # If haven't initialize the first label data 
        if is_folder_empty(cfg.active_learning.label_data_folder):
            initial_first_label_set(unlabel_data, subdata_metadata, metadata)
            print('\nSuccessfully initiated first set of label data.\nDouble check the label and run module again ...')
            sys.exit()
    
        label_data = pd.read_excel(f'{cfg.active_learning.label_data_folder.strip("/")}/{cfg.active_learning.label_data_filename}')
        
        # Start active learning
        confidence_threshold = cfg.models.spam_detection.labelling_confidence_threshold
        updated_label_data, updated_unlabel_data = start_active_training(label_data, unlabel_data, threshold=confidence_threshold)
        

if __name__ == '__main__':
    main()