import pandas as pd

# Basic EDA
def basic_eda(data):
    # 1. Any missing value ?
    print('Any missing value: ')
    print(data.isna().sum(), end='\n\n')
     
    # 2. Any duplicate messages ? 
    print('Any duplicated value: ')
    print(data.duplicated().value_counts(), end='\n\n')

    return data.isna().any().any(), data.duplicated().any()