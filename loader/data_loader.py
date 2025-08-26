import sys
import pandas as pd
import mysql.connector

from src.decorators import timer, error_log
from .config_loader import cfg


class Database:
    def __init__(self, host, port, user, password): 
        self.host = host
        self.port = port
        self.user = user
        self.password = password
    
    @error_log 
    @timer
    def connect_db(self):
        """
        Connect to target server

        Raises:
            ValueError: source unknown
        """
         
        try:
            return mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
        except mysql.connector.Error as e:
            print(f'Data Loader: Connection failed due to: {e}')
            sys.exit()
    
    
    @error_log
    @timer
    def get_metadata(self, cursor):
        """
        Return latest unlabel metadata(s)

        Args:
            cursor (mysql.connector): connector to execute query

        Returns:
            pd.DataFrame: latest subdata grouping metadata
        """
        
        cursor.execute(cfg.active_learning.subdata_metadata_query)
        metadata = pd.DataFrame(cursor.fetchall(), columns=cfg.active_learning.column_name) 
        
        finished_metadata = pd.read_excel(f'{cfg.module_log.process_log_path.files.label_record_file}')
        
        if not finished_metadata.empty:
            metadata = metadata.merge(finished_metadata, how='left', indicator=True)
            metadata = metadata[metadata._merge == 'left_only']
            metadata = metadata.drop('_merge', axis=1)
        
        return metadata