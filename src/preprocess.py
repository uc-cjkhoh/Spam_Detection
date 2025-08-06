import pandas as pd 
import ftfy
import re
import emoji  
import sys

from loader.config_loader import cfg
from . decorators import timer, error_log
 
from sklearn.preprocessing import LabelEncoder
from lingua import Language, LanguageDetectorBuilder 

_languages = [Language.ENGLISH, Language.CHINESE, Language.MALAY, Language.TAMIL]
_detector = LanguageDetectorBuilder.from_languages(*_languages).build() 

class custom_filter_regex:
    # string starting with imsi
    _imsi = '^(imsi=[0-9]+&uid=[0-9a-z]+&[a-z]=[0-9]&[a-z]+=[0-9a-z]+)$'
    
    # string start with 0#
    _international = '^([0-9]#[0-9]+#(?:\+[0-9]+|[0-9]+))$'
    
    # url link
    _url_link = '(?:^((?:https|http|)?:\/\/(www\.)?[^\s\/$.?#].[^\s]*)$|([0-9a-z]+[. :{}]{1}[0-9a-z.: {}]+))'
    
    # request_link
    _request = '(?:^ac\/.+|^reg\-req\?.+)'
    
    # all special character
    _spec_char = '[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?~`。、]'
    
    # phone number
    _phone_number = '(?:\+?60|0)1[0-46-9][\s\-]?\d{3}[\s\-]?\d{4}'
    
    # words contain number
    _no_char_mix = '(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]+'
    
    # only all chars
    _only_char = '[a-zA-Z]+'
    
    # only number
    _only_num = '[0-9]+'
    

@error_log
@timer
def text_normalize(data: pd.DataFrame):
    """
    Normalize message structure

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
      
    try:
        data['decoded_message'] = data[cfg.data.target_column].apply(ftfy.fix_text)
        data['decoded_message'] = data['decoded_message'].apply(str.strip)
        data['decoded_message'] = data['decoded_message'].apply(str.lower)
        data['decoded_message'] = data['decoded_message'].apply(lambda x: re.sub('\s+', ' ', x))
        data['decoded_message'] = data['decoded_message'].apply(lambda x: x.replace('\n', ' '))
        
        data['decoded_message'] = data['decoded_message'].apply(lambda x: emoji.replace_emoji(x, '<EMO>'))
        # data['decoded_message'] = data['decoded_message'].apply(lambda x: re.sub(custom_filter_regex._spec_char, '.', x))
        # data['decoded_message'] = data['decoded_message'].apply(lambda x: re.sub(custom_filter_regex._no_char_mix, '', x))
        data = data.drop(cfg.data.target_column, axis=1)
         
        return data
    except KeyError:
        print('Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml')
        sys.exit()
      

@error_log
@timer       
def feature_engineering(data: pd.DataFrame):
    """
    Understand message's patterns and extract them

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: data with patterns extracted
    """
    
    # how many number in message
    def find_numeric_length(message: str):
        return len(str(''.join(re.findall('\d+', message))))
     
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
     
    # find word to character ratio
    def find_word_char_ratio(message: str):
        return 
     
    # determine if message contain certain language
    def message_contain_language(message: pd.Series):
        confidence = _detector.compute_language_confidence_values(message)
        
        lgs = [cfd.language.name for cfd in confidence if cfd.value != 0]
        cfd_value = [cfd.value for cfd in confidence if cfd.value != 0]
        
        most_possible_lg = lgs[cfd_value.index(max(cfd_value))] if cfd_value != [] else 'OTHERS'
        
        return most_possible_lg
        
    try: 
        data['decoded_message_length'] = data['decoded_message'].apply(len)
        
        data['num_spec_percent'] = \
            (data['decoded_message'].apply(find_numeric_length) + \
            data['decoded_message'].apply(find_special_char_len)) / data['decoded_message_length']
        
        # data['has_url_link'] = data['decoded_message'].apply(find_url) 
        
        # data['has_phone_number'] = data['decoded_message'].apply(find_phone_number)
          
        # data['word_to_char_ratio'] = data['decoded_message'].apply(find_word_char_ratio)
          
        data['language'] = data['decoded_message'].apply(message_contain_language)
        
        label = LabelEncoder()
        data['language'] = label.fit_transform(data['language'])
        
        data = data.drop([cfg.data.target_column + '_length'], axis=1)
        return data, label.classes_
    except KeyError:
        print(f'Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml')
        sys.exit()
