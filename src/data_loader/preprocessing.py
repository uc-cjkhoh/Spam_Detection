import re
import pandas as pd
import numpy as np

from typing import List, Dict
from prefect import task
from prefect.cache_policies import NO_CACHE


class SMSTextCleaner:
    def __init__(self): 
        """ Define character to convert """
        self.char_replacements = {
            '0': 'o',
            '1': 'i',
            '3': 'e',
            '4': 'a',
            '5': 's',
            '7': 't',
            '8': 'b',
            '@': 'a',
            '$': 's',
            '!': 'i',
        }
         
        self.unicode_patterns = [
            (r'[𝐀-𝐙𝐚-𝐳]', self._normalize_bold),
            (r'[𝑨-𝒁𝒂-𝒛]', self._normalize_italic),
            (r'[𝙰-𝚉𝚊-𝚣]', self._normalize_sans),
        ]
         
        self.financial_keywords = [
            'loan', 'bank', 'interest', 'free', 'bonus', 'claim', 'voucher',
            'win', 'prize', 'cash', 'money', 'credit', 'approved', 'guaranteed'
        ]
        
        self.action_keywords = [
            'reply', 'whatsapp', 'wassap', 'pm', 'contact', 'click', 'register',
            'apply', 'join', 'call', 'sms', 'text', 'message'
        ]
        
        self.urgency_keywords = [
            'urgent', 'now', 'today', 'limited', 'hurry', 'fast', 'immediate',
            'expire', 'deadline', 'last chance', 'act now'
        ]
        
        self.suspicious_keywords = [
            'akpk', 'blacklisted', 'ctos', 'ccris', 'commitment', 'reduce',
            'consolidate', 'debt', 'refinance'
        ]
        
        self.social_media_keywords = [
            'whatsapp', 'wassap', 'telegram', 'wechat', 'line', 'viber'
        ]
    
    def _normalize_bold(self, match):
        """Convert bold unicode to normal
         
        Args:
            match (re.Match)

        Returns:
            str: replaced character
        """
        char = match.group(0)
        code = ord(char)
        if 0x1D400 <= code <= 0x1D419: 
            return chr(code - 0x1D400 + ord('A'))
        elif 0x1D41A <= code <= 0x1D433: 
            return chr(code - 0x1D41A + ord('a'))
        return char
    
    def _normalize_italic(self, match):
        """Convert italic unicode to normal

        Args:
            match (re.Match)

        Returns:
            str: replaced character
        """
        char = match.group(0)
        code = ord(char)
        if 0x1D434 <= code <= 0x1D44D:  
            return chr(code - 0x1D434 + ord('A'))
        elif 0x1D44E <= code <= 0x1D467: 
            return chr(code - 0x1D44E + ord('a'))
        return char
    
    def _normalize_sans(self, match):
        """Convert sans-serif unicode to normal

        Args:
            match (re.Match)

        Returns:
            str: replaced character
        """
        char = match.group(0)
        code = ord(char)
        if 0x1D5A0 <= code <= 0x1D5B9: 
            return chr(code - 0x1D5A0 + ord('A'))
        elif 0x1D5BA <= code <= 0x1D5D3: 
            return chr(code - 0x1D5BA + ord('a'))
        return char
    
    def remove_excessive_whitespace(self, text: str) -> str:
        """Remove leading/trailing/excessive whitespace and normalize newlines

        Args:
            text (str): raw text in string data type

        Returns:
            str: processed text
        """ 
        text = text.strip() 
        text = re.sub(r' +', ' ', text) 
        text = re.sub(r'\n+', '\n', text) 
        text = text.replace('\t', ' ')
        return text
    
    def normalize_unicode_chars(self, text: str) -> str:
        """Normalize fancy unicode characters to standard ASCII

        Args:
            text (str): text contains emoji, symbol, other special characters

        Returns:
            str: processed text
        """
        for pattern, normalizer in self.unicode_patterns:
            text = re.sub(pattern, normalizer, text)
        return text
    
    def remove_emojis(self, text: str) -> str:
        """Remove all emoji characters""" 
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def keep_emojis_count(self, text: str) -> int:
        """Count emojis (can be used as feature)"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251" 
            "]+",
            flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def normalize_obfuscated_text(self, text: str) -> str:
        """
        Normalize common spam obfuscation patterns
        e.g., 'L0an' -> 'Loan', 'Fr0m' -> 'From'
        """
        # Apply character replacements only to alphabetic contexts
        result = []
        for char in text:
            if char in self.char_replacements:
                # Check context: replace if surrounded by letters
                result.append(self.char_replacements[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def remove_special_chars(self, text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters, optionally keeping basic punctuation"""
        if keep_punctuation:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;\'-]', ' ', text)
        else:
            # Keep only alphanumeric and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text
    
    def normalize_phone_numbers(self, text: str) -> str:
        """Normalize phone number formats (Malaysian patterns)"""
        # Pattern: 60123456789 or 6 0 1 2 3 4 5 6 7 8 9
        text = re.sub(r'6\s*0\s*1\s*\d{1}\s*\d{1}\s*\d{1}\s*\d{1}\s*\d{1}\s*\d{1}\s*\d{1}', 'PHONENUMBER', text)
        # Pattern: 0123456789
        text = re.sub(r'0\d{9,10}', 'PHONENUMBER', text)
        # Pattern: +60123456789
        text = re.sub(r'\+?60\d{9,10}', 'PHONENUMBER', text)
        return text
    
    def normalize_urls(self, text: str) -> str:
        """Replace URLs with placeholder"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, 'URL', text)
        return text
    
    def normalize_currency(self, text: str) -> str:
        """Normalize currency mentions"""
        # RM patterns
        text = re.sub(r'RM\s*\d+[\d,]*', 'CURRENCY', text, flags=re.IGNORECASE)
        text = re.sub(r'rm\s*\d+[\d,]*', 'CURRENCY', text, flags=re.IGNORECASE)
        return text
    
    def convert_to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()
     
    def count_char_substitutions(self, text: str) -> int:
        """
        Count character substitutions commonly used in spam (0→O, 3→E, etc.)
        Detects patterns like 'L0an', 'Fr3e', 'B0nus'
        """
        count = 0
        # Look for digits in word contexts
        for match in re.finditer(r'\b\w*\d+\w*\b', text):
            word = match.group()
            # Check if word has letters before/after digits (likely substitution)
            if re.search(r'[a-zA-Z]\d', word) or re.search(r'\d[a-zA-Z]', word):
                count += 1
        return count
    
    def count_keyword_category(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords from a category"""
        text_lower = text.lower()
        count = 0
        for keyword in keywords:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        return count
    
    def has_fragmented_phone(self, text: str) -> int:
        """
        Detect fragmented phone numbers like '0 1 1 6 0 5 0 8 4 0 3'
        Returns 1 if found, 0 otherwise
        """
        # Pattern: single digit followed by space, repeated
        pattern = r'\b\d\s+\d\s+\d\s+\d\s+\d\s+\d'
        return 1 if re.search(pattern, text) else 0
    
    def count_social_media_mentions(self, text: str) -> int:
        """Count social media platform mentions"""
        return self.count_keyword_category(text, self.social_media_keywords)
    
    def count_excessive_newlines(self, text: str) -> int:
        """Count sequences of multiple consecutive newlines"""
        return len(re.findall(r'\n\n+', text))
    
    def count_repeated_chars(self, text: str) -> int:
        """
        Count sequences of repeated characters (3+ times)
        e.g., '!!!', '***', '😭😭😭'
        """
        # Count repeated punctuation
        punct_repeats = len(re.findall(r'([!?*=#])\1{2,}', text))
        # Count repeated emojis (any char repeated 3+ times)
        emoji_repeats = len(re.findall(r'(.)\1{2,}', text))
        return punct_repeats + emoji_repeats
    
    def has_mixed_scripts(self, text: str) -> int:
        """
        Detect mixed language scripts (Latin + CJK characters)
        Returns 1 if found, 0 otherwise
        """
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
        return 1 if (has_latin and has_cjk) else 0
    
    def count_all_caps_words(self, text: str) -> int:
        """Count words that are ALL CAPS (2+ characters)"""
        words = text.split()
        return sum(1 for word in words if len(word) >= 2 and word.isupper())
    
    def has_call_to_action(self, text: str) -> int:
        """
        Detect call-to-action phrases
        Returns count of CTA phrases found
        """
        cta_patterns = [
            r'\breply\b', r'\bpm\b', r'\bcontact\b', r'\bclick\b',
            r'\bregister\b', r'\bapply\b', r'\bjoin\b', r'\bcall\b',
            r'\bwhatsapp\b', r'\bmessage\b', r'\bsms\b'
        ]
        text_lower = text.lower()
        count = 0
        for pattern in cta_patterns:
            if re.search(pattern, text_lower):
                count += 1
        return count
    
    def count_exclamation_question(self, text: str) -> int:
        """Count excessive punctuation (multiple ! or ?)"""
        return len(re.findall(r'[!?]{2,}', text))
    
    def has_unicode_special_chars(self, text: str) -> int:
        """
        Detect special Unicode characters (bold, italic, fancy fonts)
        Returns 1 if found, 0 otherwise
        """
        # Check for mathematical alphanumeric symbols
        pattern = r'[\U0001D400-\U0001D7FF]'
        return 1 if re.search(pattern, text) else 0
    
    def count_consecutive_digits(self, text: str) -> int:
        """Count sequences of 4+ consecutive digits"""
        return len(re.findall(r'\d{4,}', text))
    
    def calculate_contact_diversity(self, text: str) -> int:
        """
        Count number of different contact methods mentioned
        (phone, whatsapp, telegram, etc.)
        """
        text_lower = text.lower()
        contact_methods = ['phone', 'whatsapp', 'wassap', 'telegram', 'wechat', 
                          'viber', 'line', 'pm', 'sms', 'call']
        count = sum(1 for method in contact_methods if method in text_lower)
        return count
    
    def extract_features(self, text: str) -> Dict:
        """Extract all features before cleaning (existing + new)"""
        features = { 
            'emoji_count': self.keep_emojis_count(text),
            'has_phone': np.where(bool(re.search(r'0\d{9,10}|6\s*0\s*1', text)), 1, 0),
            'has_currency': np.where(bool(re.search(r'rm\s*\d+', text, re.IGNORECASE)), 1, 0),
            'has_url': np.where(bool(re.search(r'http[s]?://', text)), 1, 0),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
            'length': len(text),
            'newline_count': text.count('\n'), 
            'char_substitution_count': self.count_char_substitutions(text),
            'financial_keyword_count': self.count_keyword_category(text, self.financial_keywords),
            'action_keyword_count': self.count_keyword_category(text, self.action_keywords),
            'urgency_keyword_count': self.count_keyword_category(text, self.urgency_keywords),
            'suspicious_keyword_count': self.count_keyword_category(text, self.suspicious_keywords),
            'has_fragmented_phone': self.has_fragmented_phone(text),
            'social_media_count': self.count_social_media_mentions(text),
            'excessive_newlines': self.count_excessive_newlines(text),
            'repeated_chars_count': self.count_repeated_chars(text),
            'has_mixed_scripts': self.has_mixed_scripts(text),
            'all_caps_words_count': self.count_all_caps_words(text),
            'call_to_action_count': self.has_call_to_action(text),
            'excessive_punctuation': self.count_exclamation_question(text),
            'has_unicode_special': self.has_unicode_special_chars(text),
            'consecutive_digits_count': self.count_consecutive_digits(text),
            'contact_method_diversity': self.calculate_contact_diversity(text),
        }
        return features
    
    def clean(self, text: str, remove_emojis: bool = True, normalize_numbers: bool = True, to_lowercase: bool = True) -> str:
        """Text Cleaning Pipeline

        Args:
            text (str): raw text
            remove_emojis (bool, optional): Whether to remove emojis. Defaults to True.
            normalize_numbers (bool, optional): Whether to replace phone or currency to text. Defaults to True.
            to_lowercase (bool, optional): Whether to convert text to lowercase. Defaults to True.

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
         
        text = self.remove_excessive_whitespace(text)
         
        text = self.normalize_unicode_chars(text)
         
        if remove_emojis:
            text = self.remove_emojis(text)
         
        text = self.normalize_urls(text)
         
        if normalize_numbers:
            text = self.normalize_phone_numbers(text)
            text = self.normalize_currency(text)
         
        text = self.normalize_obfuscated_text(text)
         
        text = self.remove_special_chars(text, keep_punctuation=True)
         
        text = self.remove_excessive_whitespace(text)
         
        if to_lowercase:
            text = self.convert_to_lowercase(text)
        
        return text


@task(name="Feature Engineering", cache_policy=NO_CACHE)
def feature_engineering(messages: pd.Series) -> pd.DataFrame: 
    cleaner = SMSTextCleaner()
    
    messages = messages.fillna('') 
    feature_dicts = messages.apply(cleaner.extract_features)
    feature_df = pd.DataFrame(feature_dicts.tolist())
     
    return feature_df


@task(name="Clean Text", cache_policy=NO_CACHE)
def clean_text(messages: pd.Series) -> pd.Series:
    cleaner = SMSTextCleaner()
    
    messages = messages.fillna('')    
    messages = messages.apply(
        lambda x: cleaner.clean(
            x, 
            remove_emojis=True, 
            normalize_numbers=True, 
            to_lowercase=True
        )
    ) 
    
    return messages