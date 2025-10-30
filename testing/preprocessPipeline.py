import pandas as pd 
import ftfy
import re
import emoji  

from loader.config_loader import cfg
from ..src.decorators import timer, error_log

 
class PreprocessPipeline:
    def __init__(self):
        self.regex = {
            'msg_start_with_imsi': '^(imsi=[0-9]+&uid=[0-9a-z]+&[a-z]=[0-9]&[a-z]+=[0-9a-z]+)$',
            'imsi+msisdn': '^([0-9]#[0-9]+#(?:\+[0-9]+|[0-9]+))$',
            'url_link': '(?:^((?:https|http|)?:\/\/(www\.)?[^\s\/$.?#].[^\s]*)$|([0-9a-z]+[. :{}]{1}[0-9a-z.: {}]+))',
            'request': '(?:^ac\/.+|^reg\-req\?.+)',
            'spec_char': '[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?~`。、]',
            'phone_number': '(?:\+?60|0)1[0-46-9][\s\-]?\d{3}[\s\-]?\d{4}',
            'no_char_mix': '(?=[a-zA-Z]*\d)(?=\d*[a-zA-Z])[a-zA-Z0-9]+',
            'only_char': '[a-zA-Z]+',
            'only_num': '[0-9]+'
        }
    
    @timer
    @error_log
    def text_normalize(self, message: pd.Series) -> pd.Series:
        """
        Normalize message structure

        Args:
            data (pd.DataFrame): data 

        Returns:
            pd.DataFrame: add two columns (decoded_message, decoded_message_length)
        """
        
        try:
            message = message.apply(ftfy.fix_text)
            message = message.apply(str.strip)
            # message = message.apply(str.lower)
            message = message.apply(lambda x: re.sub('\s+', ' ', x))
            message = message.apply(lambda x: x.replace('\n', ' '))
            message = message.apply(lambda x: emoji.replace_emoji(x, '<EMO>'))
            
            if cfg.data.drop_null:
                data = data.dropna()
            if cfg.data.drop_duplicates:
                data = data.drop_duplicates()
                
            # message = message.apply(lambda x: re.sub(self.regex['spec_char'], '.', x))
            # message = message.apply(lambda x: re.sub(self.regex['no_char_mix'], '', x))
        
            return data
        except KeyError:
            print('Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml')
            raise