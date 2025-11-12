import pandas as pd
import mysql.connector 

from prefect import task
from prefect.cache_policies import NO_CACHE
 
 
class Database:
    def __init__(self, cfg):
        self.cfg = cfg
        self.connector = self.initialize_db_connection()
        self.cur = self.connector.cursor()

    @task(cache_policy=NO_CACHE)
    def initialize_db_connection(self):
        try:     
            return mysql.connector.connect(
                host = self.cfg.server.host,
                port = self.cfg.server.port,
                user = self.cfg.server.user,
                password = self.cfg.server.password
            )
        except Exception as e:
            raise Exception(e)
    
    @task(cache_policy=NO_CACHE)
    def get_population_metadata(self, query, columns: list) -> pd.DataFrame:
        self.cur.execute(query)
        data =  pd.DataFrame(self.cur.fetchall(), columns=columns)
        return data 

    @task(cache_policy=NO_CACHE)
    def retrieve_subdata_by_query(self, query, columns: list) -> pd.DataFrame:
        self.cur.execute(query)
        data = pd.DataFrame(self.cur.fetchall(), columns=columns)
        
        metadata_column = self.cfg.data.metadata_column
        metadata = data[metadata_column].to_dict(orient='records')

        return data, metadata

    def get_cursor(self):
        return self.cur
     
    def close_connection(self):
        self.cur.close()
        self.connector.close()
        
        