import yaml


class Config:
    def __new__(self):
        with open('./app/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            return config
        

def get_config():
    return Config() 