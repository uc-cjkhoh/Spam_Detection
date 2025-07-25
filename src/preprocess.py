import pandas as pd
import numpy as np
import ftfy
import re
import emoji

from urllib.parse import urlparse

# Two part of preprocessing
# 1. Fix Mojibake
# 2. Feature Engineering
# 3. Remove unnecessary row(s) or column(s)
 
def fix_mojibake(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all message to utf-8 type

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
     
    try:
        data['decoded_message'] = data['message'].apply(ftfy.fix_text)
        data['decoded_message'] = data['decoded_message'].apply(str.strip)
        data['decoded_message'] = data['decoded_message'].apply(str.lower)
        data['decoded_message'] = data['decoded_message'].apply(lambda x: x.replace('\n', ' '))
        data['decoded_message'] = data['decoded_message'].apply(lambda x: emoji.replace_emoji(x, ''))
        data['decoded_message_length'] = data['message'].apply(len)
    except KeyError:
        print('Column name is not defined')
        
    return data
        

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    For all message, check if they:
        1. Has only numeric or special characters   (Done)
        2. Emotion of message by sentiment analysis
        3. Has any url link(s)
        4. Has phone number(s)                      (Done)
        5. Involve any money 
        6. Custom filter 

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data
    """
    
    # how many number in message
    def find_numeric_length(message):
        return len(str(''.join(re.findall('\d+', message))))
    
    # how many special character in message
    def find_special_char_len(message):
        return len(str(''.join(re.findall('[@_!#$%^&*()<>?/\|}{~:+-]', message))))
    
    # find if any phone number exists
    def find_phone_number(message):
        return str(re.findall('(?:\+?60|0)1[0-46-9][\s\-]?\d{3}[\s\-]?\d{4}', message))
    
    # find if there any url link(s) exists
    def find_url(message):
        return str(re.findall('(?:https|http|)?:\/\/(?:www\.)?[^\s\/$.?#].[^\s]*', message))
    
    # custom filtering
    def custom_filter(message):
        filter_1 = message[:4] == 'imsi'
        return filter_1
    
    try:
        # 1. Has only numeric or special characters
        data['all_num_special'] = np.where(
            data['message_length'] == data['decoded_message'].apply(find_numeric_length) + \
                                      data['decoded_message'].apply(find_special_char_len),
            True, False
        )
        
        # 2. Find any url link(s) exists
        data['url_link'] = data['decoded_message'].apply(find_url)
        
        # 3. Find any phone number
        data['phone_number'] = data['decoded_message'].apply(find_phone_number)
        
        # 5. Cutom Filter
        data['custom'] = data['decoded_message'].apply(custom_filter)
    except KeyError:
        print(f'Column name is not defined')
    
    return data 