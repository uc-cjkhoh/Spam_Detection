# === Standard Imports ===
import os
import sys
import pandas as pd 
import re 
import joblib 
from sqlalchemy import create_engine

from tqdm import tqdm 
from datetime import datetime
 
from src.decorators import timer, error_log
from src.llm import text_embedding

from loader.data_loader import Database
from loader.config_loader import cfg
from loader.logger_loader import logging
  
_query = 'SELECT id, current_datetime, filter_group_id, filter_group_name, result_flag, error_reason_tag, payload, is_suspicious FROM sms_spam_cd.data_tdr_spam_filter \
        WHERE YEAR(current_datetime) = {} AND \
        MONTH(current_datetime) = {} \
        AND DAY(current_datetime) = {} \
        AND HOUR(current_datetime) = {} \
        AND DAY(current_datetime) * 100 + HOUR(current_datetime) > 1615'
  

def text_normalize(data: pd.DataFrame):
    """
    Normalize message structure

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
      
    try:
        # translated = data[cfg.data.target_column] = data[cfg.data.target_column].apply(ftfy.fix_text)
        data[cfg.data.target_column] = data[cfg.data.target_column].apply(str.strip)
        # data[cfg.data.target_column] = data[cfg.data.target_column].apply(str.lower)
        data[cfg.data.target_column] = data[cfg.data.target_column].apply(lambda x: re.sub(r'\s+', ' ', x))
        data[cfg.data.target_column] = data[cfg.data.target_column].apply(lambda x: x.replace('\n', ' '))
        # data[cfg.data.target_column] = data[cfg.data.target_column].apply(lambda x: emoji.replace_emoji(x, '<EMO>'))
         
        if cfg.data.drop_null:
            data = data.dropna()
        if cfg.data.drop_duplicates:
            data = data.drop_duplicates()
            
        # data[cfg.data.target_column] = data[cfg.data.target_column].apply(lambda x: re.sub(custom_filter_regex._spec_char, '.', x))
        # data[cfg.data.target_column] = data[cfg.data.target_column].apply(lambda x: re.sub(custom_filter_regex._no_char_mix, '', x))
    
        return data
    except KeyError:
        print('Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml')
        sys.exit()
        

def get_metadata(cursor):
    """
    Return latest unlabel metadata(s)

    Args:
        cursor (mysql.connector): connector to execute query

    Returns:
        pd.DataFrame: latest subdata grouping metadata
    """
    
    cursor.execute(cfg.active_learning.subdata_metadata_query)
    metadata = pd.DataFrame(cursor.fetchall(), columns=cfg.active_learning.column_name) 
    
    return metadata
  
  
def spam_detection(model, message):  
    spam_label = model.predict(message)
    confidence_score = model.predict_proba(message)
    return spam_label, confidence_score.max(axis=1)
 
  
@error_log
@timer
def main(): 
    database = Database(
        host=cfg.server.host,
        port=cfg.server.port,
        user=cfg.server.user,
        password=cfg.server.password
    )
    
    connector = database.connect_db()
    cur = connector.cursor()
    
    all_metadata = get_metadata(cur) 
      
    for metadata in tqdm(all_metadata.to_numpy()):
        print(f'Ingesting subdata: {metadata}')
        query = _query.format(*metadata)
        
        cur.execute(query)
        unlabel_data = pd.DataFrame(cur.fetchall(), columns=['id', 'current_datetime', 'filter_group_id', 'filter_group_name', 'result_flag', 'error_reason_tag', 'payload', 'is_suspicious'])
        
        model = joblib.load('models/SGDClassifier.joblib')
        spam_label, confidence_score = spam_detection(model, text_embedding(unlabel_data['payload']))
        unlabel_data['spam_label'] = spam_label
        unlabel_data['confidence_score'] = confidence_score
        # normalized_data = text_normalize(unlabel_data[['message']].copy()) 
        
        # unlabel_data['translated_data'] = translated_data
        # unlabel_data['normalized_data'] = normalized_data
        dt = str(min(unlabel_data['current_datetime']))
        
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
        
        # filename = dt.strftime('%Y%m%d%H') + '.xlsx'
        # to_folder = 'testing'
        # unlabel_data.to_excel(os.path.join(to_folder, filename)) 
        
        # Rename columns BEFORE trying to insert
        unlabel_data = unlabel_data.rename(columns={
            'spam_label': 'ml_spam_label',
            'confidence_score': 'ml_confidence_score'
        })
         
        # Create engine and insert
        engine = create_engine(
            f'mysql+pymysql://{cfg.server.user}:{cfg.server.password}@{cfg.server.host}:{cfg.server.port}/sms_spam_cd'
        )
        
        unlabel_data.to_sql(
            name='ml_spam_result',
            con=engine, 
            schema='sms_spam_cd',
            if_exists='append',
            index=False
        )
        
        # Close engine
        engine.dispose()
                  

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        logging.error(e, exc_info=True)
    finally:
        logging.info('===== End of Execution =====')
