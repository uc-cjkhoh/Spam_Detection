# for all the sub-data
#   Question 1: what is the accuracy we could conclude with data been fitted ?
#   Question 2: what is the accuracy we could conclude with data not been fitted ? 

import os
import sys
import joblib
import pandas as pd
 
from scipy import stats
from sklearn.metrics import classification_report

from src.model import Custom_Models
from src.llm import text_embedding 
from loader.config_loader import cfg

_target_model = f'models/{sys.argv[1]}'


class Data:
    def __init__(self, file_path: str):
        self.data = pd.read_excel(file_path)
    
    def get_random_sample(self, frac=0.2):
        return self.data.sample(frac, replace=False)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
 

def t_test(population_data: pd.DataFrame): 
    random_sample = population_data.sample(
        frac=cfg.hypothesis_testing.overall_perf_test.sample_size,
        replace=False
    )
        
    # hypothesis testing based on confidence score 
    _, confidence_score = model_prediction(random_sample[cfg.data.target_column]) 
    _, p_value = stats.ttest_1samp(confidence_score, popmean=cfg.hypothesis_testing.overall_perf_test.target_performance)
        
    return p_value
        

def main():
    label_folder = cfg.active_learning.label_data_folder
    unlabel_folder = cfg.active_learning.unlabel_data_folder
    
    all_folder = [label_folder, unlabel_folder]
    model = Custom_Models(joblib.load(_target_model))

    for is_unlabel, folder_path in enumerate(all_folder):
        files = os.listdir(folder_path)
        for _file in files:
            filepath = os.path.join(folder_path, _file)
            data = Data(filepath)
            random_data = data.get_random_sample()
            
            if is_unlabel:
                pass
            else:
                # test model with metrics (accuracy, precision, recall)
                y_true = random_data[cfg.data.target_coulmn + '_label']
                y_pred = model.predict(
                    text_embedding(random_data[cfg.data.target_column])
                )
                
                classification_report(y_true, y_pred)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(e)
    finally:
        logging.info("End of Execution")