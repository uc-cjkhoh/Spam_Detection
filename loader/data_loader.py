import sys
import pandas as pd
import mysql.connector

from src.util import update_unfinish_metadata
from src.decorators import timer, error_log
from . config_loader import cfg


class Database:
    def __init__(self, source):
        self.source = source 
    
    @error_log 
    @timer
    def connect_db(self):
        """
        Connect to target server

        Raises:
            ValueError: source unknown
        """
        
        if self.source is None:
            raise ValueError("Source is not define, make source is define in config.yaml")
        else:
            try:
                return mysql.connector.connect(
                    host=cfg.server.host,
                    port=cfg.server.port,
                    user=cfg.server.user,
                    password=cfg.server.password
                )
            except mysql.connector.Error as e:
                print(f'Data Loader: Connection failed due to: {e}')
                sys.exit()
    
    @error_log
    @timer
    def get_metadata(self, cursor):
        """
        Return latest subdata grouping metadata

        Args:
            cursor (mysql.connector): connector to execute query

        Returns:
            pd.DataFrame: latest subdata grouping metadata
        """
        
        cursor.execute(cfg.active_learning.subdata_metadata_query)
        metadata = pd.DataFrame(cursor.fetchall(), columns=cfg.active_learning.column_name)
        update_unfinish_metadata(metadata) 
        return metadata