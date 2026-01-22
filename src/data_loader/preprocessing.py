import re
import pandas as pd
import numpy as np
from typing import List, Dict
import unicodedata


class SMSTextCleaner:
    """
    Comprehensive text cleaner for SMS spam detection
    Handles special characters, unicode variations, obfuscation patterns, and emojis
    """
    
    def __init__(self):
        # Common spam obfuscation patterns
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
        
        # Unicode variations to normalize
        self.unicode_patterns = [
            # Bold/Italic/Sans-serif mathematical alphanumeric symbols
            (r'[𝐀-𝐙𝐚-𝐳]', self._normalize_bold),
            (r'[𝑨-𝒁𝒂-𝒛]', self._normalize_italic),
            (r'[𝙰-𝚉𝚊-𝚣]', self._normalize_sans),
        ]
    
    def _normalize_bold(self, match):
        """Convert bold unicode to normal"""
        char = match.group(0)
        code = ord(char)
        if 0x1D400 <= code <= 0x1D419:  # Bold capitals
            return chr(code - 0x1D400 + ord('A'))
        elif 0x1D41A <= code <= 0x1D433:  # Bold lowercase
            return chr(code - 0x1D41A + ord('a'))
        return char
    
    def _normalize_italic(self, match):
        """Convert italic unicode to normal"""
        char = match.group(0)
        code = ord(char)
        if 0x1D434 <= code <= 0x1D44D:  # Italic capitals
            return chr(code - 0x1D434 + ord('A'))
        elif 0x1D44E <= code <= 0x1D467:  # Italic lowercase
            return chr(code - 0x1D44E + ord('a'))
        return char
    
    def _normalize_sans(self, match):
        """Convert sans-serif unicode to normal"""
        char = match.group(0)
        code = ord(char)
        if 0x1D5A0 <= code <= 0x1D5B9:  # Sans capitals
            return chr(code - 0x1D5A0 + ord('A'))
        elif 0x1D5BA <= code <= 0x1D5D3:  # Sans lowercase
            return chr(code - 0x1D5BA + ord('a'))
        return char
    
    def remove_excessive_whitespace(self, text: str) -> str:
        """Remove leading/trailing/excessive whitespace and normalize newlines"""
        # Remove leading/trailing whitespace
        text = text.strip()
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        return text
    
    def normalize_unicode_chars(self, text: str) -> str:
        """Normalize fancy unicode characters to standard ASCII"""
        for pattern, normalizer in self.unicode_patterns:
            text = re.sub(pattern, normalizer, text)
        return text
    
    def remove_emojis(self, text: str) -> str:
        """Remove all emoji characters"""
        # Emoji patterns
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
    
    def extract_features(self, text: str) -> Dict:
        """Extract useful features before cleaning"""
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
        }
        return features
    
    def clean(self, text: str, remove_emojis: bool = True, normalize_numbers: bool = True, to_lowercase: bool = True) -> str:
        """
        Main cleaning pipeline
        
        Parameters:
        -----------
        text: str - Input SMS text
        remove_emojis: bool - Whether to remove emojis
        normalize_numbers: bool - Whether to replace phone/currency with tokens
        to_lowercase: bool - Whether to convert to lowercase
        
        Returns:
        --------
        str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Remove excessive whitespace
        text = self.remove_excessive_whitespace(text)
        
        # Step 2: Normalize unicode fancy characters
        text = self.normalize_unicode_chars(text)
        
        # Step 3: Remove or keep emojis
        if remove_emojis:
            text = self.remove_emojis(text)
        
        # Step 4: Normalize URLs
        text = self.normalize_urls(text)
        
        # Step 5: Normalize numbers if requested
        if normalize_numbers:
            text = self.normalize_phone_numbers(text)
            text = self.normalize_currency(text)
        
        # Step 6: Normalize obfuscated text
        text = self.normalize_obfuscated_text(text)
        
        # Step 7: Remove special characters (keep basic punctuation)
        text = self.remove_special_chars(text, keep_punctuation=True)
        
        # Step 8: Clean up whitespace again after transformations
        text = self.remove_excessive_whitespace(text)
        
        # Step 9: Convert to lowercase if requested
        if to_lowercase:
            text = self.convert_to_lowercase(text)
        
        return text


# Example usage
def get_normalized_messages(df, target_column, extract_features: bool = True): 
    df[target_column] = df[target_column].fillna('')
    
    # Initialize cleaner
    cleaner = SMSTextCleaner()
    
    # Extract features before cleaning (if requested)
    if extract_features: 
        feature_dicts = df[target_column].apply(cleaner.extract_features)
        feature_df = pd.DataFrame(feature_dicts.tolist())
        
        # Add features to dataframe
        for col in feature_df.columns:
            df[f'feature_{col}'] = feature_df[col]
    
    # Clean text 
    df[target_column] = df[target_column].apply(
        lambda x: cleaner.clean(
            x, 
            remove_emojis=True, 
            normalize_numbers=True, 
            to_lowercase=True
        )
    ) 
    
    return df

 