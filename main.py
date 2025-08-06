
import loader.data_loader
import src.preprocess
import src.model
import src.eda
import src.util
import src.active_training

from loader.config_loader import cfg
from loader.logger_loader import logging
 
import pandas as pd
from datetime import datetime 


if __name__ == '__main__':
    ### Connect to database
    database = loader.data_loader.Database('mysql').connect_db()
     
    ### Get a cursor and execute query in config file
    cur = database.cursor()
    
    ### Get subdata metadata for active learning iteration
    cur.execute(cfg.active_learning.subdata_metadata_query)
    subdata_metadata = pd.DataFrame(cur.fetchall(), columns=cfg.active_learning.column_name)
    subdata_metadata = src.util.check_learning_process(subdata_metadata)
        
    ### Get data
    for profile in subdata_metadata.to_numpy():
        query = cfg.data.query.format(*profile)
        logging.info('Learning subdata of year: {}, month: {}, day: {}, hour: {}'.format(*profile))
        
        cur.execute(query)
        data = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
        
        ### 1. Preprocess data 
        # Try the best to normalize messages
        data = src.preprocess.text_normalize(data.copy()) 
        
        ### 2. EDA
        # Check data statistic
        # src.eda.basic_eda(data.copy(), title='Raw Data')
        if cfg.data.drop_null:
            data = data.dropna()
        if cfg.data.drop_duplicates:
            data = data.drop_duplicates()
        
        ### 3. Feature Engineering
        data, _lgs = src.preprocess.feature_engineering(data.copy())
        
        # check the statistic of new dataset
        # src.eda.basic_eda(data.copy(), title='Formatted Data', access_mode='a') 
        
        # 4. Select only string data 
        string_data = data.select_dtypes(include=['object'])
        
        ### Prepare label and unlabel data 
        # If haven't initialize the first label data
        labelled_data_folder = './labelled_data'
        if src.util.is_folder_empty(labelled_data_folder):
            for col in string_data.columns:
                result = src.model.train_model(string_data[col], column_name=col)
                data = pd.concat([data, result], axis=1)
                
            # save result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f'./labelled_data/{timestamp}.xlsx'
            
            with pd.ExcelWriter(filename) as writer:
                for group_name, group_df in data.groupby('language'):
                    group_df.to_excel(writer, sheet_name=str(_lgs[group_name]), index=False)
                
        ### Start active learning
        # confidence_threshold = cfg.models.spam_detection.labelling_confidence_threshold
        # label_data = data.groupby(by='language')