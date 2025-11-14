import pandas as pd  
import re
import emoji  

from prefect import task 
from prefect.cache_policies import NO_CACHE
 
 
class Regex:
    msg_start_with_imsi = r'^(imsi=[0-9]+&uid=[0-9a-z]+&[a-z]=[0-9]&[a-z]+=[0-9a-z]+)$'
    imsi_msisdn = r'^([0-9]#[0-9]+#(?:\+[0-9]+|[0-9]+))$'
    url_link = r'(?:^((?:https|http|)?:\/\/(www\.)?[^\s\/$.?#].[^\s]*)$|([0-9a-z]+[. :{}]{1}[0-9a-z.: {}]+))'
    request = r'(?:^ac\/.+|^reg\-req\?.+)'
    spec_char = r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?~`。、]'
    phone_number = r'(?:\+?60|0)1[0-46-9][\s\-]?\d{3}[\s\-]?\d{4}'
    no_char_mix = r'(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]+'
    only_char = r'[a-zA-Z]+'
    only_num = r'[0-9]+'
        
@task(cache_policy=NO_CACHE)
def text_normalize(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Normalize message structure

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
    
    try:
        message = data[target_column]
        # message = message.apply(ftfy.fix_text)
        message = message.apply(str.strip)
        # message = message.apply(str.lower)
        message = message.apply(lambda x: re.sub(r'\s+', ' ', x))
        message = message.apply(lambda x: x.replace(r'\n', ' '))
        message = message.apply(lambda x: emoji.replace_emoji(x, '<EMO>'))
        
        data[target_column] = message
        return data
    except KeyError:
        print('Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml')
        raise
        