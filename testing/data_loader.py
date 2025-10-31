import pandas as pd
import mysql.connector
import logging
 
 
class Database:
    def __init__(self, cfg):
        self.cfg = cfg
        self.connector = self.initialize_db_connection()
        self.cur = self.connector.cursor()
     
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
        
    def retrieve_by_query(self, query) -> pd.DataFrame:
        self.cur.execute(query)
        return pd.DataFrame(self.cur.fetchall())
         
    def close_connection(self):
        self.cur.close()
        self.connector.close()