import pandas as pd
import sys

from . decorators import error_log, timer 

@error_log
@timer
def start_active_training(label_data: pd.DataFrame, unlabel_data: pd.DataFrame, threshold: float):
    print(label_data.head())
    print(unlabel_data.head())
    print(threshold)
    
    sys.exit()
    
    return None, None