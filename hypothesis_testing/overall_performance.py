# for all the sub-data
#   Question 1: what is the accuracy we could conclude with data been fitted ?
#   Question 2: what is the accuracy we could conclude with data not been fitted ? 

import os
import joblib
import pandas as pd
 
from scipy import stats
from src.llm import text_embedding

from loader.data_loader import Database
from loader.config_loader import cfg

_target_model = r'models/'
def model_prediction(message: pd.Series):
    model = joblib.load(_target_model)
    text_vector = text_embedding(message)
    
    return model.predict(text_vector)
    

def main():
    database = Database(
        host=cfg.server.host,
        port=cfg.server.port,
        user=cfg.server.user,
        password=cfg.server.password      
    )
    
    connector = database.connect_db()
    cur = connector.cursor()
    
    label_metadata = pd.excel(cfg.module_log.process_log_path.files.label_record_file)
    unlabel_metadata = pd.excel(cfg.module_log.process_log_path.files.unlabel_record_file)
    
    for metadata in label_metadata:
        query = cfg.data.query.format(*metadata)
        cur.execute(query)
        
        data = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
        data['prediction'] = model_prediction(data[cfg.data.target_name])
        


if __name__ == "__main__":
    main()