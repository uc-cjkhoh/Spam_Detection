# === Standard Imports ===
import pandas as pd
import numpy as np
import sys
import os

from tqdm import tqdm
from sklearn.linear_model import SGDClassifier

# === Project Imports ===
from src.preprocess import text_normalize
from src.util import setup_directory_and_file, update_metadata
from src.decorators import timer, error_log 
from src.llm import initial_labeling
from src.model import Custom_Models
# from src.eda import basic_eda

from loader.data_loader import Database
from loader.config_loader import cfg
from loader.logger_loader import logging
  
  
@error_log
@timer
def main():
    setup_directory_and_file()
    
    database = Database(
        host=cfg.server.host,
        port=cfg.server.port,
        user=cfg.server.user,
        password=cfg.server.password
    )
    
    connector = database.connect_db()
    cur = connector.cursor()
    
    all_metadata = database.get_metadata(cur)
    update_metadata(all_metadata)
    
    subdata_log_dt = ('{}/' * len(cfg.active_learning.column_name)).strip('/')
    
    sgd_model = SGDClassifier(loss='modified_huber', random_state=42)
    _model = Custom_Models(sgd_model)
    
    for metadata in tqdm(all_metadata.to_numpy()):
        finished_metadatas = pd.read_excel(cfg.module_log.process_log_path.files.label_record_file)
        
        if len(finished_metadatas) != 0 and np.any(np.all(finished_metadatas.to_numpy() == metadata, axis=1)):
            logging.info(f'Skipping labelled metadata {subdata_log_dt.format(*metadata)}')
            continue
        
        query = cfg.data.query.format(*metadata)
        logging.info(f'Loading subdata from: {subdata_log_dt.format(*metadata)}')
        
        cur.execute(query)
        unlabel_data = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
        unlabel_data = text_normalize(unlabel_data.copy()) 
         
        if len(pd.read_excel(cfg.module_log.process_log_path.files.label_record_file)) == 0:
            label_data = initial_labeling(unlabel_data[cfg.data.target_column])  
            
            label_folder = cfg.active_learning.label_data_folder
            label_filename = f'{len(os.listdir(label_folder))}.xlsx' 
            label_filepath = os.path.join(label_folder, label_filename)
            
            label_data.to_excel(
                label_filepath, 
                index=False
            )
            
            update_metadata()
            logging.info('\nSuccessfully initiated first set of label data.\nDouble check each label and run module again ...')
            return 0
        else:
            label_folder = cfg.active_learning.label_data_folder
            label_files = os.listdir(label_folder)
            
            label_data = None
            for _file in label_files:
                if label_data is None:
                    label_data = pd.read_excel(os.path.join(label_folder, _file))
                else:
                    label_data = pd.concat([label_data, pd.read_excel(os.path.join(label_folder, _file))])                        
                    
            _model.start_active_training(
                label_data, 
                unlabel_data, 
                threshold=cfg.models.spam_detection.labelling_confidence_threshold
            ) 
    
            update_metadata()
        

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        logging.error(e, exc_info=True)
    finally:
        logging.info('===== End of Execution =====')