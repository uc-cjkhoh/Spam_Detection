import pandas as pd
import mysql.connector 

from prefect import task
from prefect.cache_policies import NO_CACHE
 
 
class Database:
    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connector = self.initialize_db_connection()
        self.cur = self.connector.cursor()
 
    def initialize_db_connection(self):
        try:     
            return mysql.connector.connect(
                host = self.host,
                port = self.port,
                user = self.user,
                password = self.password
            )
        except Exception as e:
            raise Exception(e)
     
    @task(cache_policy=NO_CACHE)
    def run_query(self, query, columns: list) -> pd.DataFrame:
        self.cur.execute(query)
        data = pd.DataFrame(self.cur.fetchall(), columns=columns)
        return data

    def get_cursor(self):
        return self.cur
     
    def close_connection(self):
        try:
            if self.cur:
                self.cur.close()
        except Exception as e:
            raise Exception(f"Warning: Failed to close cursor: {e}")
        
        try:
            if self.connector:
                self.connector.close()
        except Exception as e:
            raise Exception(f"Warning: Failed to close connector: {e}")
        
        