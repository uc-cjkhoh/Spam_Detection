import os 
import yaml

from addict import Dict

_config_path = './configs/config.yaml'

if not os.path.exists(_config_path):
    raise FileNotFoundError(f'Config file is not found in {_config_path}')

with open(_config_path, 'r') as f:
    cfg = Dict(yaml.load(f, Loader=yaml.FullLoader))