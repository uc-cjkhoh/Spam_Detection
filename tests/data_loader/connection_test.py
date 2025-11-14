import pandas as pd
import pytest

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.data_loader.connection import Database


@pytest.fixture
def mock_cfg():
    mock_cfg = MagicMock()
    mock_cfg.server.host = '10.168.51.196'
    mock_cfg.server.port = 3306
    mock_cfg.server.user = 'unified'
    mock_cfg.server.password = 'unified'
    mock_cfg.data.metadata_column = 'id'
    
    return mock_cfg

@pytest.fixture
def test_mock_db(mock_cfg):
    """test database connection"""
    with patch('mysql.connector.connect') as mock_connect:
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor 
        
        db = Database(mock_cfg)
        yield db
        db.close_connection()
        
def test_run_query(test_mock_db):
    """
    test if cursor execute query once 
    test if the returned datatype is pd.DataFrame
    test if the output length is the same as input length
    """
    query = "select col1, col2, col3 from database"
    columns = ['example1', 'example2', 'example3']
    mock_data = [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9)
    ]
    
    test_mock_db.cur.fetchall.return_value = mock_data 
    result = test_mock_db.run_query(query, columns)
    
    test_mock_db.cur.execute.assert_called_once_with(query)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(mock_data)


def test_get_cursor(test_mock_db):
    """test if returned cursor is the same with existing cursor"""
    assert test_mock_db.cur == test_mock_db.get_cursor()
    
    
def test_close_connection(test_mock_db):
    """test if the close function get called"""
    test_mock_db.close_connection()
    test_mock_db.cur.close.assert_called_once()
    test_mock_db.connector.close.assert_called_once()