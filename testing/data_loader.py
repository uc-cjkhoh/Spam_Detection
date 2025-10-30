import mysql.connector

from loader.decorators import timer, error_log
from loader.config_loader import cfg
 
class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            return mysql.connector.connect(
                host = cfg.server.host,
                port = cfg.server.port,
                user = cfg.server.user,
                password = cfg.server.password
            )
    
@error_log 
@timer
def get_connector():
    return Database()
        
