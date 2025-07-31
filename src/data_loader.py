import os
import yaml
import mysql.connector

from addict import Dict

class Database:
    def __init__(self, source, config_path='./configs/config.yaml'):
        self.source = source
        self.config_path = config_path
        self.cfg = self.load_config()    
    
    def load_config(self):
        """
        Load Configuration File

        Raises:
            FileNotFoundError: If configuration file's path is not found

        Returns:
            dict: configuration in config.yaml 
        """
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f'Config file is not found in {self.config_path}')
        else:
            with open(r'./configs/config.yaml') as f:
                return Dict(yaml.load(f, Loader=yaml.FullLoader))
    
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
                    host=self.cfg.server.host,
                    port=self.cfg.server.port,
                    user=self.cfg.server.user,
                    password=self.cfg.server.password
                )
            except mysql.connector.Error as e:
                print(f'Connection failed due to: {e}')
                