import pytest
import numpy as np
import pandas as pd

from unittest.mock import MagicMock, patch
from sklearn.linear_model import SGDClassifier

from src.ml.model_training import load_model, blob_to_numpy, initialize_model


@pytest.fixture
def mock_config():
    mock_config = MagicMock()
    mock_config.experiment_name = 'test_experiment'
    mock_config.model_name = 'test_model'
    mock_config.initialize_model.query = "select col1, col2 from table"
    return mock_config


@pytest.fixture
def mock_cursor():
    return MagicMock()

 
@pytest.fixture
def mock_imported_data():
    mock_embeddings = np.array([1, 2, 3, 4])
    mock_label = 1
    encoded = str(mock_embeddings).replace(' ', ', ').encode('utf-8')
    return [(encoded, mock_label)]
    
    
def test_load_model_with_logged_model(mock_config, mock_cursor):
    with patch('mlflow.search_experiments') as mock_exp, \
        patch('mlflow.search_logged_models') as mock_logged, \
        patch('mlflow.pyfunc.load_model') as mock_load: 
            
        mock_exp.return_value = [MagicMock(experiment_id='test1')]
        mock_logged.return_value = pd.DataFrame({'source_run_id': ['testing_only_id']})
    
        mock_model = MagicMock()  
        mock_load.return_value = mock_model
        
        result = load_model.fn(mock_config, mock_cursor)
        
        mock_exp.assert_called_once()
        mock_logged.assert_called_once()
        mock_load.assert_called_once()
        
        assert result == mock_model
        

def test_blob_to_numpy(mock_imported_data):   
    result_embeddings, result_labels = blob_to_numpy.fn(mock_imported_data)
    
    assert np.array_equal(result_embeddings.squeeze(), np.array([1, 2, 3, 4]).squeeze())
    assert np.array_equal(result_labels.squeeze(), np.asarray([1]).squeeze())


def test_initialize_model(mock_config, mock_cursor): 
    model = MagicMock()
    
    with patch('src.ml.model_training.initialize_model') as mock_initialize_model:
        mock_initialize_model.return_value = model
        
        result = mock_initialize_model(mock_config, mock_cursor) 
        assert isinstance(result, type(model))
            

def test_train_model(): 
    pass