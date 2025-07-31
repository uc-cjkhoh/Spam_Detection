
import src.data_loader
import src.preprocess
import src.model
import src.eda

import os 
import yaml
import pandas as pd
pd.set_option('display.max_rows', None)

from addict import Dict


if __name__ == '__main__':
    config_path = './configs/config.yaml'

    ### Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file is not found in {config_path}')
    else:
        with open(r'./configs/config.yaml') as f:
            cfg = Dict(yaml.load(f, Loader=yaml.FullLoader))

    ### Connect to database
    database = src.data_loader.Database('mysql', config_path).connect_db()
     
    ### Get a cursor and execute query in config file
    cur = database.cursor()
    cur.execute(cfg.data.query)
    
    ### Get data
    data = pd.DataFrame(cur.fetchall(), columns=cfg.data.column_name)
     
    ### Preprocess data 
    # 1. Try the best to normalize messages
    data = src.preprocess.text_normalize(data.copy(), filter=True) 
    
    ### EDA
    # 1. Check any missing value or duplicate value
    src.eda.basic_eda(data.copy())
    if cfg.data.drop_null:
        data = data.dropna()
    if cfg.data.drop_duplicates:
        data = data.drop_duplicates()
    
    ### 2. Feature Engineering
    data = src.preprocess.feature_engineering(data.copy())
    
    ### 3. Data normalization
    numeric_data = data.select_dtypes(exclude=['object'])
    minmax_scaled, robust_scaled, standard_scaled = src.preprocess.data_normalization(numeric_data.copy())
    
    ### 4. Model Training
    # model, result = src.model.train_model({
    #     'min_max_scaled': minmax_scaled,
    #     'robust_scaled': robust_scaled,
    #     'standard_scaled': standard_scaled
    # })
    
    # embedded_message = src.model.text_embedding(data['decoded_' + cfg.data.target_column])
    
    # print(
    #     data[data['custom'] != 1][[
    #         'filtered_message',
    #         'filtered_message_length',
    #         'filtered_message_words',
    #         'decoded_message', 
    #         'decoded_message_length', 
    #         'number_of_words',
    #         'num_spec_percent',
    #         'is_chinese',
    #         'is_english',
    #         'is_malay',
    #         'is_tamil',
    #         'is_other',
    #         'has_url_link',
    #         'has_phone_number'
    #     ]].reset_index(drop=True)
    # )
    
    