import pytest
import pandas as pd

from src.data_loader.preprocessing import get_normalized_messages

# 1. To test where the received parameters is pd.DataFrame and string data type
# 2. To test the data length is the same as preprocessing
# 3. To test the return data type is list 
def test_get_normalized_messages():
    sample_df = pd.DataFrame({
        'message': [
            '   Hello',
            'Hi\nthere',
            'Multiple\n\nlines \t here 😂'
        ]
    })
    
    expect = [
        'Hello',
        'Hi there',
        'Multiple lines here <EMO>'
    ]
    
    result = get_normalized_messages(sample_df, target_column='message')
    
    assert result == expect
    assert isinstance(result, list)
    

def test_get_normalized_messages_invalid_column():
    sample_df = pd.DataFrame({'text': ['Hello']})
    
    with pytest.raises(KeyError, match='Invalid column'): 
        get_normalized_messages(sample_df, target_column='message')