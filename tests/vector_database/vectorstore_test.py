import pytest
import numpy as np

from unittest.mock import MagicMock, patch
from prefect.testing.utilities import prefect_test_harness

from src.vector_database.vectorstore import VectorStore


@pytest.fixture(autouse=True)
def disable_prefect_task():
    with prefect_test_harness():
        yield


@pytest.fixture
def mock_config():
    mock_config = MagicMock()
    mock_config.directory = 'directory'
    mock_config.filename = 'file.txt'
    mock_config.embedding = 'embedding_model'
    return mock_config
    

@pytest.fixture
def mock_vectordb(mock_config):
    return VectorStore(
        mock_config.directory,
        mock_config.filename,
        mock_config.embedding
    )
    
    
@pytest.fixture
def mock_faiss_index():
    return MagicMock()


def test_check_any_existing_vectorstore_with_existing_db(mock_vectordb, mock_config): 
    filepath = 'directory/file.txt'
    is_exists = True
    
    with patch("os.path.join", return_value=filepath) as path_join, \
         patch("os.path.exists", return_value=is_exists) as path_exist, \
         patch("src.vector_database.vectorstore.FAISS.load_local") as load_local:
             
        mock_vectordb._check_any_existing_vectorstore()
        
        path_join.assert_called_once_with(mock_config.directory, mock_config.filename)
        path_exist.assert_called_once_with(filepath)
        load_local.assert_called_once()
        

def test_check_any_existing_vectorstore_with_no_existing_db(mock_vectordb, mock_config): 
    filepath = None
    is_exists = False
    
    with patch("os.path.join", return_value=filepath) as path_join, \
         patch("os.path.exists", return_value=is_exists) as path_exist:
             
        mock_vectordb._check_any_existing_vectorstore()
        
        path_join.assert_called_once_with(mock_config.directory, mock_config.filename)
        path_exist.assert_called_once_with(filepath)
        

def test_write_to_vectorstore_with_valid_index(mock_vectordb, mock_faiss_index):
    mock_text_embedding_pair = MagicMock()
    mock_embedding_model = MagicMock()
    mock_metadatas = MagicMock()
    
    mock_vectordb.index = mock_faiss_index
    
    mock_vectordb.write_to_vectorstore(
        mock_text_embedding_pair, 
        mock_embedding_model, 
        mock_metadatas
    )

    mock_faiss_index.add_embeddings.assert_called_once()


def test_write_to_vectorstore_with_no_index(mock_vectordb):
    mock_text_embedding_pair = MagicMock()
    mock_embedding_model = MagicMock()
    mock_metadatas = MagicMock()
    
    mock_vectordb.index = None 
     
    with patch("src.vector_database.vectorstore.FAISS.from_embeddings") as from_embeddings:
        mock_vectordb.write_to_vectorstore(
            mock_text_embedding_pair,
            mock_embedding_model,
            mock_metadatas
        )
        from_embeddings.assert_called_once()


def test_save_with_existing_index(mock_vectordb, mock_faiss_index):
    mock_vectordb.index = mock_faiss_index
    
    mock_vectordb.save()
    
    mock_faiss_index.save_local.assert_called_once_with(
        folder_path=mock_vectordb.directory,
        index_name=mock_vectordb.filename
    )
    

def test_save_with_no_index(mock_vectordb):
    mock_vectordb.index = None
    
    with pytest.raises(ValueError, match='Cannot save empty index'):
        mock_vectordb.save()
    
    