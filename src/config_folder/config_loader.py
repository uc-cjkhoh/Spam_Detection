import os 
import yaml
import errno
  
from prefect import task
from prefect.cache_policies import NO_CACHE

from data_validation.configs_validation.validate_config_loader import ProjectConfig


class ConfigLoader:  
    def __new__(cls):
        try:
            _config_path = r'./configs/config.yaml' 
            with open(_config_path, 'r') as f:
                config = yaml.safe_load(f)
                return ProjectConfig(**config)
                    
        except FileNotFoundError as e:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _config_path)
    

@task(name="Load YAML Configuration", cache_policy=NO_CACHE)     
def get_config():
    """
    Load config.yaml document

    Returns:
        Dict: All configuration settings in dictionary format
    """
    
    return ConfigLoader()

