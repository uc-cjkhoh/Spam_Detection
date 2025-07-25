import os 
import yaml
import pandas as pd
pd.set_option('display.max_rows', None)

import src.data_loader
import src.preprocess
import src.eda

if __name__ == '__main__':
    config_path = './configs/config.yaml'

    ## Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file is not found in {config_path}')
    else:
        with open(r'./configs/config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    ## Connect to database
    database = src.data_loader.Database('mysql', config_path).connect_db()
     
    ## Get a cursor and execute query in config file
    cur = database.cursor()
    cur.execute(cfg['data']['query'])
    
    ## Get data
    data = pd.DataFrame(cur.fetchall(), columns=cfg['data']['column_name'])
    
    ## Basic EDA
    
    
    ## Preprocess data 
    # 1. Try the best to convert mojibake to normal utf-8
    data = src.preprocess.fix_mojibake(data.copy())
    
    # 2. Feature Engineering
    data = src.preprocess.feature_engineering(data.copy())
    

    print(
        data[
            ~((data['all_num_special'] == True) | (data['custom'] == True))
        ][['decoded_message', 'url_link', 'custom']].reset_index(drop=True))
    # Train data
    
    # Save result and model