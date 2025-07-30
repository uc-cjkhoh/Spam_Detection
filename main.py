import os 
import yaml
import pandas as pd
import src.data_loader
import src.preprocess
# import src.model
import src.eda

pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    config_path = './configs/config.yaml'

    ### Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file is not found in {config_path}')
    else:
        with open(r'./configs/config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    ### Connect to database
    database = src.data_loader.Database('mysql', config_path).connect_db()
     
    ### Get a cursor and execute query in config file
    cur = database.cursor()
    cur.execute(cfg['data']['query'])
    
    ### Get data
    data = pd.DataFrame(cur.fetchall(), columns=cfg['data']['column_name'])
     
    ### Preprocess data 
    # 1. Try the best to normalize messages
    data = src.preprocess.text_normalize(data.copy())
    
    ### EDA
    # 1. Missing value or Duplicate value
    has_missing, has_duplicates = src.eda.basic_eda(data)
    
    if has_missing:
        data = data.dropna()
    
    if has_duplicates:
        data = data.drop_duplicates()
    
    ### 2. Feature Engineering
    data = src.preprocess.feature_engineering(data.copy())
     
    # filtered_data = data[~((data['has_no_word_char'] == True) | (data['custom'] == True))][
    #     ['decoded_message', 'decoded_message_length', 'has_url_link', 'custom']
    # ].reset_index(drop=True)
    
    print(
        data[data['custom'] == 0][[
            'decoded_message', 
            'decoded_message_length', 
            'num_spec_percent', 
            'has_url_link', 
            'has_chinese',
            'has_english',
            'has_malay',
            'has_phone_number', 
            'custom' # filter patterns discovered when developing
            
        ]].reset_index(drop=True)
    )
    
    