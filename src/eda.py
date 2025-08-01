import pandas as pd
import os
from . decorators import timer, error_log

# Basic EDA
@error_log
@timer
def basic_eda(data, title=None, access_mode='w'):
    eda_output_file = r"./eda_result.txt"
    
    with open(eda_output_file, access_mode) as f:
        if title:
            f.write(f'==========> {title} <==========\n\n')
        else: 
            f.write('=================================\n\n')
        
        # Describe
        f.write('Data Types: \n')
        for name in data.columns:
            f.write(f'{name}: {type(data[name][0])}\n')
        f.write('\n\n')
        
        f.write('Numeric Data Statistic: \n')
        f.write(str(data.select_dtypes(exclude=['object']).describe()))
        f.write('\n\n')
        
        # Any missing value ?
        f.write('Any missing value: \n')
        f.write(str(data.isna().sum()))
        f.write('\n\n')
        
        # Any duplicate messages ? 
        f.write('Any duplicated value: \n')
        f.write(str(data.duplicated().value_counts()))
        f.write('\n\n')
        
        f.write('=================================\n')