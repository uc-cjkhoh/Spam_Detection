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

 
def test_load_model_with_logged_model(mock_config, mock_cursor):
    with patch('mlflow.search_experiments') as mock_exp, patch('mlflow.search_logged_models') as mock_logged, patch('mlflow.pyfunc.load_model') as mock_load: 
        mock_exp.return_value = [MagicMock(experiment_id='test1')]
        mock_logged.return_value = pd.DataFrame({'source_run_id': ['testing_only_id']})
    
        mock_model = MagicMock()  
        mock_load.return_value = mock_model
        
        result = load_model(mock_config, mock_cursor)
        
        mock_exp.assert_called_once()
        mock_logged.assert_called_once()
        mock_load.assert_called_once_with("run:/testing_only_id/test_model")
        
        assert result == mock_model
        

def test_blob_to_numpy(): 
    mock_embeddings = np.random.rand(1, 1024).tolist()
    mock_label = 1
    encoded = str(mock_embeddings).encode('utf-8')
    mock_imported_data = [(encoded, mock_label)]
    
    result_embeddings, result_labels = blob_to_numpy(mock_imported_data)
    
    assert np.array_equal(result_embeddings.squeeze(), np.asarray(mock_embeddings).squeeze())
    assert np.array_equal(result_labels.squeeze(), np.asarray([1]).squeeze())


def test_initialize_model(mock_config, mock_cursor):
    with patch('blob_to_numpy') as to_numpy:
        model = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 2),
            (3 ,4)
        ]
        
        result = initialize_model(mock_config, mock_cursor)
        mock_cursor.execute.assert_called_once()
        to_numpy.assert_called_once_with()
            



def test_train_model():
    pass