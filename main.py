
import src.data_loader
import src.preprocess
import src.model
import src.eda
import src.util
import src.active_training

from src.config_loader import cfg
 
import pandas as pd
pd.set_option('display.max_rows', None)

from datetime import datetime 


if __name__ == '__main__':
    ### Connect to database
    database = src.data_loader.Database('mysql').connect_db()
     
    ### Get a cursor and execute query in config file
    cur = database.cursor()
    cur.execute(cfg.data.query)
    
    ### Get data
    data = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
    
    ### Preprocess data 
    # Try the best to normalize messages
    data = src.preprocess.text_normalize(data.copy()) 
    
    ### EDA
    # Check data statistic
    src.eda.basic_eda(data.copy(), title='Raw Data')
    if cfg.data.drop_null:
        data = data.dropna()
    if cfg.data.drop_duplicates:
        data = data.drop_duplicates()
    
    ### Feature Engineering
    data, _lgs = src.preprocess.feature_engineering(data.copy())
    
    # check the statistic of new dataset
    src.eda.basic_eda(data.copy(), title='Formatted Data', access_mode='a') 
    
    ### Select all message columns
    string_data = data.select_dtypes(include=['object'])
     
    ### If haven't, initialize First Labelling
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
            
    ### active learning process
    confidence_threshold = cfg.models.spam_detection.labelling_confidence_threshold
    label_data = data.groupby(by='language')
    print(label_data)