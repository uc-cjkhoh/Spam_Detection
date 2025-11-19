# import pytest
# import pandas as pd
# import numpy as np

# from unittest.mock import MagicMock, patch
# from sentence_transformers import SentenceTransformer

# from src.data_loader.embedding import Embedding

# @pytest.fixture
# def mock_obj():
#     mock_embedding = MagicMock()
#     mock_embedding.model_name = 'testing'
    
#     mock_obj = Embedding(mock_embedding)
    
#     mock_model = SentenceTransformer(mock_embedding.model_name, trust_remote_code=True)
#     mock_obj.model = mock_model 
    
#     return mock_obj


# def test_embed_message(mock_obj):
#     """
#     test if the returned result is a np.ndarray instance
#     test if the returned result has same length as input data
#     """
#     mock_messages = pd.DataFrame(
#         {'message': ['this is an example message', 'this is another one', 'and another', 'and another']}
#     )
#     mock_target_column = 'message' 
#     mock_batch_size = 4 
    
#     result = mock_obj.embed_message(mock_messages, mock_target_column, mock_batch_size)

#     assert isinstance(result, np.ndarray)
#     assert len(result) == len(mock_messages)


# def test_get_embeddings(mock_config):
#     assert isinstance(mock_config.model, SentenceTransformer)