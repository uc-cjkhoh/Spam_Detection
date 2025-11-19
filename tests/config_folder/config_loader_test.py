from addict import Dict 
from src.config_folder.config_loader import get_config

def test_get_config():
    assert isinstance(get_config(), Dict)