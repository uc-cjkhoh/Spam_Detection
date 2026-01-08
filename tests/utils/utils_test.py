import pytest
import numpy as np
import pandas as pd
 
from unittest.mock import patch, MagicMock  
from prefect.testing.utilities import prefect_test_harness
 
from src.utils.util import setup_core_instances, create_required_folder_file, update_metadata, blob_to_numpy


@pytest.fixture(autouse=True)
def disable_prefect_task():
    with prefect_test_harness():
        yield


@pytest.fixture
def mock_mysql(): 
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]

    mock_connector = MagicMock()
    mock_connector.cursor.return_value = mock_cursor

    with patch("mysql.connector.connect", return_value=mock_connector):
        yield mock_connector, mock_cursor


@pytest.fixture
def mock_imported_data():
    mock_embeddings = np.array([1, 2, 3, 4])
    mock_label = 1
    encoded = str(mock_embeddings).replace(' ', ', ').encode('utf-8')
    return [(encoded, mock_label)]

 
def test_setup_core_instances(mock_mysql):
    config = MagicMock()
    config.models.text_embedding.model_name = "fake-model"
    config.vectorstore.directory = "dir"
    config.vectorstore.filename = "file"

    mock_connector, mock_cursor = mock_mysql

    # Prepare mocks
    mock_db = MagicMock()
    mock_emb = MagicMock()
    mock_vs = MagicMock()

    with patch('mysql.connector.connect', return_value=mock_connector), \
        patch("src.data_loader.connection.Database", return_value=mock_db), \
        patch("src.utils.util.HuggingFaceEmbeddings", return_value=mock_emb), \
        patch("src.utils.util.VectorStore", return_value=mock_vs):

        db, emb, vs = setup_core_instances(config)

        assert db.connector is mock_connector
        assert db.cur is mock_cursor
        
        assert emb is mock_emb
        assert vs is mock_vs
    
 
def test_create_required_folder_file_with_no_existing():
    config = MagicMock() 
    config.progress_log.folder = '/folder'
    config.vectorstore.directory = '/vectorstore_folder'
    config.metadata.column_name = ['col1', 'col2']
    
    config.progress_log.files.finished = 'finished_file.xlsx'
    config.progress_log.files.unfinished = 'unfinished_file.xlsx'
    
    dirs = MagicMock()
    is_file = MagicMock(True)
    
    with patch('src.utils.util.os.makedirs') as makedirs, \
        patch('src.utils.util.os.path.isfile', return_value=False) as isfile, \
        patch('pandas.DataFrame.to_excel') as to_excel:
        
        result = create_required_folder_file(config)
        
        makedirs.assert_any_call("/folder", exist_ok=True)
        makedirs.assert_any_call("/vectorstore_folder", exist_ok=True)
        assert makedirs.call_count == 2
        
        isfile.assert_any_call('finished_file.xlsx')
        isfile.assert_any_call('unfinished_file.xlsx')
        assert isfile.call_count == 2
        
        to_excel.assert_any_call('finished_file.xlsx', index=False)
        to_excel.assert_any_call('unfinished_file.xlsx', index=False)
        assert to_excel.call_count == 2
    
    
def test_create_required_folder_file_with_existing():
    config = MagicMock() 
    config.progress_log.folder = '/folder'
    config.vectorstore.directory = '/vectorstore_folder'
    config.metadata.column_name = ['col1', 'col2']
    
    config.progress_log.files.finished = 'finished_file.xlsx'
    config.progress_log.files.unfinished = 'unfinished_file.xlsx'
    
    dirs = MagicMock()
    is_file = MagicMock(True)
    
    with patch('src.utils.util.os.makedirs') as makedirs, \
        patch('src.utils.util.os.path.isfile', return_value=True) as isfile, \
        patch('pandas.DataFrame.to_excel') as to_excel:
        
        result = create_required_folder_file(config)
        
        makedirs.assert_any_call("/folder", exist_ok=True)
        makedirs.assert_any_call("/vectorstore_folder", exist_ok=True)
        assert makedirs.call_count == 2
        
        isfile.assert_any_call('finished_file.xlsx')
        isfile.assert_any_call('unfinished_file.xlsx')
        assert isfile.call_count == 2  
            

def test_not_none_update_metadata():
    config = MagicMock()
    config.progress_log.files.unfinished = 'unfinished.xlsx'
    
    mock_metadata = pd.DataFrame({
        'meta1': [1],
        'meta2': [2],
        'meta3': [3]
    })

    with patch('pandas.DataFrame.to_excel') as to_excel:
        update_metadata(config, mock_metadata)
        to_excel.assert_called_once_with('unfinished.xlsx', index=False)
    
    
def test_none_update_metadata():            
    config = MagicMock()
    config.progress_log.files.finished = 'finished.xlsx'
    config.progress_log.files.unfinished = 'unfinished.xlsx'
    
    fake_finished = MagicMock()
    fake_unfinished = MagicMock()
    fake_unfinished.iloc = MagicMock()
    fake_unfinished.iloc.__getitem__.return_value.to_frame.return_value.T = 'ROW'
     
    fake_updated_finished = MagicMock()
    fake_updated_unfinished = MagicMock()
     
    with patch('pandas.DataFrame.to_excel') as to_excel, \
        patch('pandas.read_excel') as read_excel, \
        patch('pandas.concat', return_value=fake_updated_finished) as concat:
        
        read_excel.side_effect = [fake_finished, fake_unfinished]
 
        update_metadata(config)
        
        read_excel.assert_any_call('finished.xlsx')
        read_excel.assert_any_call('unfinished.xlsx')
        assert read_excel.call_count == 2
        
        concat.assert_called_once()
        
        fake_unfinished.drop.assert_called_once_with(index=0)
          
        fake_updated_finished.to_excel.assert_called_once_with('finished.xlsx', index=False)
        
        fake_updated_unfinished = fake_unfinished.drop.return_value
        fake_updated_unfinished.to_excel.assert_any_call('unfinished.xlsx', index=False)
   

def test_blob_to_numpy(mock_imported_data):   
    result_embeddings, result_labels = blob_to_numpy(mock_imported_data)
    
    assert np.array_equal(result_embeddings.squeeze(), np.array([1, 2, 3, 4]).squeeze())
    assert np.array_equal(result_labels.squeeze(), np.asarray([1]).squeeze())
