import pandas as pd
import os
from . decorators import timer

# Basic EDA
@timer
def basic_eda(data):
    eda_output_file = r"./eda_result.txt"
    
    with open(eda_output_file, 'w') as f:
        # 1. Any missing value ?
        f.write('Any missing value: \n')
        f.write(str(data.isna().sum()))
        f.write('\n\n')
        
        # 2. Any duplicate messages ? 
        f.write('Any duplicated value: \n')
        f.write(str(data.duplicated().value_counts()))
