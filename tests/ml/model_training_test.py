import pytest

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from unittest.mock import MagicMock, patch
from src.ml.model_training import SGD, XGBoost


@pytest.fixture
def mock_experiments():
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = 1234 
    mock_experiments = [mock_experiment]
    return mock_experiments
        
@pytest.fixture
def mock_model_list():
    model_list = [MagicMock()]
    return model_list

@pytest.fixture
def mock_config():
    mock_config = MagicMock()
    mock_config.experiment_name = 'test_experiment'
    mock_config.model_name = 'test_model'
    return mock_config
 
def test_xgboost(mock_experiments, mock_model_list, mock_config):
    with patch('src.ml.model_training.mlflow.search_experiments', return_value=mock_experiments), \
         patch('src.ml.model_training.mlflow.search_logged_models', return_value=mock_model_list):
        
        model = XGBoost(
            experiment_name=mock_config.experiment_name,
            model_name=mock_config.model_name
        )
        
        assert model.experiment == mock_experiments[0].experiment_id
        assert model.model_list[0] == mock_model_list[0]
        assert isinstance(model.model, XGBClassifier)
        
def test_sgd(mock_experiments, mock_model_list, mock_config):
    with patch('src.ml.model_training.mlflow.search_experiments', return_value=mock_experiments), \
         patch('src.ml.model_training.mlflow.search_logged_models', return_value=mock_model_list):
        
        model = SGD(
            experiment_name=mock_config.experiment_name,
            model_name=mock_config.model_name
        )
        
        assert model.experiment == mock_experiments[0].experiment_id
        assert model.model_list[0] == mock_model_list[0]
        assert isinstance(model.model, SGDClassifier)