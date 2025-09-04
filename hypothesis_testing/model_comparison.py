import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import joblib
import argparse
import pandas as pd
 
from tqdm import tqdm
from scipy import stats
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.model import Custom_Models
from src.llm import text_embedding 
from src.decorators import timer, error_log

from loader.config_loader import cfg
from loader.logger_loader import logging
 
 
class Data:
    def __init__(self, file_path: str):
        self.data = pd.read_excel(file_path)
    
    def get_random_sample(self, size=0.2):
        if isinstance(size, float):
            n = int(len(self.data) * size)
        else:
            n = size
        
        return self.data.sample(n, replace=False)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data) 


class Model(Custom_Models):
    def __init__(self, model, model_name):
        super().__init__(model)
        self.model_name = model_name 

    def predict(self, data):
        return self.model.predict(data)

    def get_model_name(self):
        return self.model_name


@timer
@error_log
def get_valid_args(argv: list):
    if len(argv) == 0:
        raise KeyError('No model(s) was specify, please run with `python model_comparison.py [-m|-d] [model(s) name|directory name]`')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', nargs='*')
    parser.add_argument('-d', '--directory') 

    known_arg, unknown_arg = parser.parse_known_args()
    if len(unknown_arg) > 0:     
        raise KeyError(f'Found unknown arg: {unknown_arg}')
    
    if known_arg.model is not None and known_arg.directory is not None:
        raise KeyError('Too many argument: Use only either -m or -d')
    
    return known_arg
 
 
@timer
@error_log
def main():
    arg = get_valid_args(sys.argv[1:])

    arg_name, arg_value = None, None
    for name, value in vars(arg).items():
        if value is not None:
            arg_name = name
            arg_value = value
        else:
            raise ValueError(f{'Missing value for argument "{name}"'})
            
    # load model(s)
    model_list = []
    if arg_name == 'model':
        for i, model in enumerate(arg_value):
            try:
                model_list.append(
                    Model(
                        model=joblib.load(os.path.join(cfg.models.save_model_to, model)),
                        model_name=f'model_{i}'
                    )
                )
            except FileNotFoundError as e:
                logging.error(f"Can't find model at path: {model}")
    else:
        for i, model_file in enumerate(os.listdir(arg_value)):
            model_list.append(
                Model(
                    model=joblib.load(os.path.join(cfg.models.save_model_to.folder, model_file)),
                    model_name=f'model_{i}'
                )
            )
    
    # load data
    all_data_filepath = os.listdir(cfg.active_learning.label_data_folder)

    result_columns = ['Model', 'Subdata', 'No_Test', 'Accuracy', 'Precision', 'Recall', 'F1']
    result_table = pd.DataFrame(columns=result_columns) 
    
    for _filepath in tqdm(all_data_filepath):
        data = Data(os.path.join(cfg.active_learning.label_data_folder, _filepath))
         
        for no_of_test in tqdm(range(cfg.hypothesis_testing.model_comparison.no_of_test)):
            random_sample = data.get_random_sample()
            random_message, y_true = random_sample[cfg.data.target_column], random_sample[cfg.data.target_column + '_label']
            message_vector = text_embedding(random_message)
            
            for model in model_list:
                y_pred = model.predict(message_vector)
                
                result_table = pd.concat([
                    result_table, 
                    pd.DataFrame([[
                        model.get_model_name(),
                        _file,
                        'test' + str(no_of_test+1),
                        accuracy_score(y_true, y_pred),
                        precision_score(y_true, y_pred),
                        recall_score(y_true, y_pred),
                        f1_score(y_true, y_pred)
                    ]], columns=result_columns)
                ])
        
    # output the result in order based on performance
    result_table.to_excel(f'hypothesis_testing/{datetime.now().strftime("%Y%m%d_%H_%M_%S")}.xlsx')
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        logging.error(e, exc_info=True)
    finally:
        logging.info('End of Execution')