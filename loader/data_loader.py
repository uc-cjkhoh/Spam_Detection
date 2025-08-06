import sys
import mysql.connector
from . config_loader import cfg


class Database:
    def __init__(self, source):
        self.source = source 
    
    def connect_db(self):
        """
        Connect to target server

        Raises:
            ValueError: source unknown
        """
        
        if self.source is None:
            raise ValueError("Data Loader: Source is not define, make source is define in config.yaml")
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
     