import os 
import yaml
import errno

from prefect import task
from prefect.cache_policies import NO_CACHE
from addict import Dict 


class ConfigLoader:  
    def __new__(cls):
        _config_path = r'./configs/config.yaml'
        if os.path.exists(_config_path):
            try:
                with open(_config_path, 'r') as f:
                    return Dict(yaml.load(f, Loader=yaml.FullLoader))
            except FileNotFoundError as e:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _config_path)
    
    
@task(cache_policy=NO_CACHE)
def get_config():
    """
    Load config.yaml document

    Returns:
        Dict: All configuration settings in dictionary format
    """
    return ConfigLoader()

