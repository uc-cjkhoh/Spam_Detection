import pandas as pd
import numpy as np
import ftfy
import re
import emoji 
import yaml
import os
import sys
 
from . decorators import timer

from addict import Dict
from lingua import Language, LanguageDetectorBuilder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

_languages = [Language.ENGLISH, Language.CHINESE, Language.MALAY, Language.TAMIL]
_detector = LanguageDetectorBuilder.from_languages(*_languages).build()
_min_max = MinMaxScaler()
_robust_scaler = RobustScaler()
_standard_scaler = StandardScaler()

_config_path = r'./configs/config.yaml'
if not os.path.exists(_config_path):
    raise FileNotFoundError(f'Config file is not found in {_config_path}')
with open(_config_path) as f:
    cfg = Dict(yaml.load(f, Loader=yaml.FullLoader))


class custom_filter_regex:
    # string starting with imsi
    _imsi = '^(imsi=[0-9]+&uid=[0-9a-z]+&[a-z]=[0-9]&[a-z]+=[0-9a-z]+)$'
    
    # string start with 0#
    _international = '^([0-9]#[0-9]+#(?:\+[0-9]+|[0-9]+))$'
    
    # url link
    _url_link = '(?:https|http|)?:\/\/(?:www\.)?[^\s\/$.?#].[^\s]*'
    
    # request_link
    _request = '(?:^ac\/.+|^reg\-req\?.+)'
    
    # all special character
    _spec_char = '[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?~`。、]'
    
    # phone number
    _phone_number = '(?:\+?60|0)1[0-46-9][\s\-]?\d{3}[\s\-]?\d{4}'
    
    
@timer
def text_normalize(data: pd.DataFrame, filter=None):
    """
    Normalize message structure

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
     
    def filter_message(message: str):
        patterns = [
            ("imsi",          custom_filter_regex._imsi),
            ("international", custom_filter_regex._international),
            ("url_link",      custom_filter_regex._url_link),
            ("spec_char",     custom_filter_regex._spec_char),
            ("phone_number",  custom_filter_regex._phone_number),
            ("all_number",    '\d+'),
            ("white_space",   '\s+')
        ]
        
        temp_message = message
        for name, regex in patterns:
            temp_message = re.sub(regex, ' ', temp_message)
        
        return temp_message.strip()
            
    
    try:
        data['decoded_message'] = data[cfg.data.target_column].apply(ftfy.fix_text)
        data['decoded_message'] = data['decoded_message'].apply(str.strip)
        data['decoded_message'] = data['decoded_message'].apply(str.lower)
        data['decoded_message'] = data['decoded_message'].apply(lambda x: x.replace('\n', ' '))
        data['decoded_message'] = data['decoded_message'].apply(lambda x: emoji.replace_emoji(x, ''))
        
        if filter:
            data['filtered_message'] = data['decoded_message'].apply(filter_message)
            return data
        else: 
            return data
    except KeyError:
        print('(Preprocessing.py, KeyError) Column name is not defined, check if column_name and payload_column is the same in ./configs/config.yaml')
        sys.exit()
      
      
@timer       
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
    
    # find number of words
    def find_number_of_words(message: str):
        message = message.strip()
        if not message:
            return 0 
        
        return len(message.split(' '))
    
    # how many special character in message
    def find_special_char_len(message: str):
        return len(str(''.join(re.findall(custom_filter_regex._spec_char, message))))
    
    # find if any phone number exists
    def find_phone_number(message: str):
        phone_number = re.findall(custom_filter_regex._phone_number, message)
        return 1 if phone_number else 0
    
    # find if there any url link(s) exists
    def find_url(message: str):
        link = re.findall(custom_filter_regex._url_link, message)
        return 1 if link else 0
    
    # custom filtering
    def custom_filter(message: str):
        filter_1 = bool(re.findall(custom_filter_regex._imsi, message))
        filter_2 = bool(re.findall(custom_filter_regex._international, message)) 
        filter_3 = bool(re.findall(custom_filter_regex._request, message))
        return filter_1 or filter_2 or filter_3
    
    # determine if message contain certain language
    def message_contain_language(messages: pd.Series):
        temp = []
        for message in messages:
            confidence = _detector.compute_language_confidence_values(message)
            
            lgs = [cfd.language.name for cfd in confidence if cfd.value != 0]
            cfd_value = [cfd.value for cfd in confidence if cfd.value != 0]
            
            is_chinese = cfd_value[lgs.index('CHINESE')] if 'CHINESE' in lgs else 0
            is_english = cfd_value[lgs.index('ENGLISH')] if 'ENGLISH' in lgs else 0
            is_malay = cfd_value[lgs.index('MALAY')] if 'MALAY' in lgs else 0
            is_tamil = cfd_value[lgs.index('TAMIL')] if 'TAMIL' in lgs else 0
            is_other = 1 if not lgs else 0
            
            temp.append([is_chinese, is_english, is_malay, is_tamil, is_other])
            
        return np.array(temp)
        
    try:
        if 'filtered_message' in data.columns:
            data['filtered_message_length'] = data['filtered_message'].apply(len)
            data['filtered_message_words'] = data['filtered_message'].apply(find_number_of_words) 
        
        data['decoded_message_length'] = data['decoded_message'].apply(len)
        
        data['number_of_words'] = data['decoded_message'].apply(find_number_of_words)
        
        data['num_spec_percent'] = \
            (data['decoded_message'].apply(find_numeric_length) + \
            data['decoded_message'].apply(find_special_char_len)) / data['decoded_message_length']
        
        data['has_url_link'] = data['decoded_message'].apply(find_url) 
        
        data['has_phone_number'] = data['decoded_message'].apply(find_phone_number)
         
        data['custom'] = data['decoded_message'].apply(custom_filter).astype(int)
         
        data[['is_chinese', 'is_english', 'is_malay', 'is_tamil', 'is_other']] = \
            message_contain_language(data['decoded_message'])
        
        return data 
    except KeyError:
        print(f'(Preprocessing.py, KeyError) Column name is not defined')
        sys.exit()


@timer
def data_normalization(data: pd.DataFrame, scaler=None):
    if scaler == 'min_max':
        return _min_max.fit_transform(data)
    elif scaler == 'robust':
        return _robust_scaler.fit_transform(data)
    elif scaler == 'standard':
        return _standard_scaler.fit_transform(data)
    elif scaler == None:
        return _min_max.fit_transform(data), _robust_scaler.fit_transform(data), _standard_scaler.fit_transform(data)
    else:
        raise ValueError(
            f'(Preprocessing.py, ValueError) Unknown scaler: `{scaler}` given. \
            Should be `min_max`, `robust`, `standard` or None return all scalers.'
        )
        