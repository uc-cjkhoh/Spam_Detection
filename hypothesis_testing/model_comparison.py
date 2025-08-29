import os
import sys
import joblib
import argparse
import pandas as pd
 
from scipy import stats
from datetime import datetime
from sklearn.metrics import classification_report

from src.model import Custom_Models
from src.llm import text_embedding 
from src.decorators import timer, error_log

from loader.config_loader import cfg
from loader.logger_loader import logging

log_path = f'{cfg.module_log.general_log_path.folder}/model_comparison_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filenae=log_path
)

class Data:
    def __init__(self, file_path: str):
        self.data = pd.read_excel(file_path)
    
    def get_random_sample(self, frac=0.2):
        return self.data.sample(frac, replace=False)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data) 

class Model:
    def __init__(self, model_name, model):
        self.model_name = model_name
        self.model = model

    
def get_valid_args(argv: list):
    if len(argv) == 0:
        raise KeyError('No model(s) was specify, please run with `python model_comparison.py [-m|-d] [model(s) name|directory name]`')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='*')
    parser.add_argument('-d', '--directory', nargs=1)

    known_arg, unknown_arg = parser.parse_known_args()
    if len(unknown_arg) > 0:     
        raise KeyError(f'Found unknown arg: {unknown_arg}')
    
    if known_arg.model is not None and known_arg.directory is not None:
        raise KeyError('Too many argument: Use only either -m or -d')
    
    return known_arg
    
 
def main():
    arg = get_valid_args(sys.argv[1:])

    arg_name, arg_value = None
``  for name, value in vars(arg).items():
        if value is not None:
            arg_name = name
            arg_value = value
            
    # load model(s)
    model_list = []
    if arg_name == 'model':
        for i, model in enumerate(arg_value):
            try:
                model_list.append(
                    Model(
                        model_name=f'model_{i}',
                        model=joblib.load(model)
                    )
                )
            except FileNotFoundError as e:
                logging.error(f"Can't find model at path: {model}")
    else:
        pass
    
    # load data
    
    # random sampling
    
    # For all model loaded, compare their performance towards the randomed sample
    
    # output the result in order based on performance
    
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        logging.error(e, exc_info=True)
    finally:
        logging.info('End of Execution')