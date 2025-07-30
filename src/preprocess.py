import pandas as pd
import numpy as np
import ftfy
import re
import emoji 

from urllib.parse import urlparse 
from lingua import Language, LanguageDetectorBuilder
from sklearn.preprocessing import MinMaxScaler

languages = [Language.ENGLISH, Language.CHINESE, Language.MALAY]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

# Two part of preprocessing
# 1. Fix Mojibake
# 2. Feature Engineering
# 3. Remove unnecessary row(s) or column(s)

class custom_filter_regex:
    # string starting with imsi
    _imsi = '^(imsi=[0-9]+&uid=[0-9a-z]+&[a-z]=[0-9]&[a-z]+=[0-9a-z]+)$'
    
    # string start with 0#
    _international = '^([0-9]#[0-9]+#(?:\+[0-9]+|[0-9]+))$'
    
    # url link
    _url_link = '(?:https|http|)?:\/\/(?:www\.)?[^\s\/$.?#].[^\s]*'
    
 
def text_normalize(data: pd.DataFrame):
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

        return data
    except KeyError:
        print('Column name is not defined')
         
        
def feature_engineering(data: pd.DataFrame):
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
        pd.DataFrame: data with extra engineered columns
    """
    
    # how many number in message
    def find_numeric_length(message: str):
        return len(str(''.join(re.findall('\d+', message))))
    
    # how many special character in message
    def find_special_char_len(message: str):
        return len(str(''.join(re.findall('[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?~`]', message))))
    
    # find if any phone number exists
    def find_phone_number(message: str):
        phone_number = re.findall('(?:\+?60|0)1[0-46-9][\s\-]?\d{3}[\s\-]?\d{4}', message)
        return 1 if phone_number else 0
    
    # find if there any url link(s) exists
    def find_url(message: str):
        link = re.findall(custom_filter_regex._url_link, message)
        return 1 if link else 0
    
    # custom filtering
    def custom_filter(message: str):
        filter_1 = bool(re.findall(custom_filter_regex._imsi, message))
        filter_2 = bool(re.findall(custom_filter_regex._international, message)) 
        return filter_1 or filter_2
    
    # determine if message contain certain language
    def message_contain_language(messages: pd.Series):
        temp = []
        for message in messages:
            confidence = detector.compute_language_confidence_values(message)
            
            lgs = [cfd.language.name for cfd in confidence if cfd.value != 0]
            cfd_value = [cfd.value for cfd in confidence if cfd.value != 0]
            
            has_chinese = cfd_value[lgs.index('CHINESE')] if 'CHINESE' in lgs else 0
            has_english = cfd_value[lgs.index('ENGLISH')] if 'ENGLISH' in lgs else 0
            has_malay = cfd_value[lgs.index('MALAY')] if 'MALAY' in lgs else 0
            other = 1 if not lgs else 0
            
            temp.append([has_chinese, has_english, has_malay, other])
            
        return np.array(temp)
        
    try:
        data['decoded_message_length'] = data['decoded_message'].apply(len)
        
        data['num_spec_percent'] = \
            (data['decoded_message'].apply(find_numeric_length) + \
            data['decoded_message'].apply(find_special_char_len)) / data['decoded_message_length']
        
        data['has_url_link'] = data['decoded_message'].apply(find_url) 
        
        data['has_phone_number'] = data['decoded_message'].apply(find_phone_number)
         
        data['custom'] = data['decoded_message'].apply(custom_filter).astype(int)
         
        data[['has_chinese', 'has_english', 'has_malay', 'other']] = \
            message_contain_language(data['decoded_message'])
        
        return data 
    except KeyError:
        print(f'Column name is not defined')


def data_normalization(data: pd.DataFrame):
    