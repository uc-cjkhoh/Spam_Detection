import pandas as pd

from . decorators import error_log, timer 

@error_log
@timer
def start_active_training(label_data: pd.DataFrame, unlabel_data: pd.DataFrame, confidence_threshold: float):
    pass