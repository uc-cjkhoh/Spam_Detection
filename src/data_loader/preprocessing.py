import pandas as pd  
import re
import emoji  

from prefect import task
from prefect.cache_policies import NO_CACHE 

@task(name='Text Normalization', cache_policy=NO_CACHE)        
def get_normalized_messages(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Normalize message structure

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
    
    try:
        message = data[target_column] 
        message = message.apply(str.strip) 
        message = message.apply(lambda x: re.sub(r'\s+', ' ', x))
        message = message.apply(lambda x: x.replace(r'\n+', ' '))
        message = message.apply(lambda x: emoji.replace_emoji(x, '<EMO>')) 
        return message.to_list()
    except KeyError:
        raise KeyError('Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml') 
    
 