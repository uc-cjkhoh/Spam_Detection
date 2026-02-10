import time
import requests
import numpy as np
import pandas as pd 
 
from tqdm import tqdm
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import StatementError, CompileError

from src.config_folder.config_loader import get_config
from deploy.app.schema import RetrieveData, ClassifyRequest


def get_data(params: RetrieveData) -> pd.Series:
    """Get payloads from MySQL

    Args:
        engine (Engine): sqlalchemy engine
        query (str): sql query to run

    Returns:
        pd.Series: payloads
    """ 
    
    try:
        engine = params.engine
        query = params.query
    
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn) 
            return df
        
    except StatementError as e:
        print(f'Failed to load data due to {e}')
        raise
    except CompileError as e:
        print(f'Failed to load data due to {e}')
        raise


def send_request(params: ClassifyRequest) -> tuple[np.ndarray, np.ndarray]: 
    """Send request to API

    Args:
        api_uri (str): API uri to test
        payloads (list): payloads

    Returns:
        tuple[np.ndarray, np.ndarray]: model result, model confidence score
    """
    
    api_uri = params.api_uri
    data = params.data
    rate = params.rate
    
    payloads = data.loc[:, 'payload'].to_list()
    ids = data.loc[:, 'id'].to_list()
     
    for i in tqdm(range(0, len(payloads), rate)):
        data = {'id': ids[i:i+rate], 'payload': payloads[i:i+rate]}
        _ = requests.post(api_uri, json=data).json() 
      

def main(): 
    config = get_config()
    
    connection_url = URL.create(
        drivername='mysql+pymysql',
        host=config.database.host,
        port=config.database.port,
        username=config.database.user,
        password=config.database.password,
        database=config.database.table_schema
    )
             
    engine = create_engine(connection_url)
     
    data = get_data(RetrieveData(engine=engine, query=config.stratified_sampling)) 
    
    start = time.perf_counter()  
    send_request(ClassifyRequest(api_uri='http://10.168.49.12:7654/classify', data=data, rate=32))
    end = time.perf_counter()
    
    print(f'Runtime: {end - start:.6f} seconds') 
     
    engine.dispose()
    
    
if __name__ == '__main__': 
    main()