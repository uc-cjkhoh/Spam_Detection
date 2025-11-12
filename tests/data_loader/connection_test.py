from src.data_loader.connection import Database

import pytest
from unittest.mock import MagicMock, patch
from src.data_loader.connection import Database


@pytest.fixture
def mock_db():
    # Create a mock configuration
    mock_cfg = MagicMock()
    mock_cfg.server.host = '10.168.51.196'
    mock_cfg.server.port = 3306
    mock_cfg.server.user = 'unified'
    mock_cfg.server.password = 'unified'
    mock_cfg.data.metadata_column = 'id'  # Example metadata column

    # Initialize the Database object with the mock configuration
    db = Database(mock_cfg)
    
    # Mock the cursor and connection
    db.connector = MagicMock()
    db.cur = db.connector.cursor.return_value
    return db

def test_initialize_db_connection(mock_db):
    # Act
    connection = mock_db.initialize_db_connection()

    # Assert
    assert connection is not None
    mock_db.connector.cursor.assert_called_once()

def test_get_population_metadata(mock_db):
    # Arrange
    query = "SELECT * FROM test_table"
    columns = ['id', 'name', 'value']
    mock_db.cur.fetchall.return_value = [(1, 'test1', 100), (2, 'test2', 200)]

    # Act
    result = mock_db.get_population_metadata(query, columns)

    # Assert
    assert result.shape == (2, 3)  # Check the shape of the DataFrame
    assert list(result.columns) == columns  # Check the columns
    assert result['id'].tolist() == [1, 2]  # Check the data
    assert result['name'].tolist() == ['test1', 'test2']
    assert result['value'].tolist() == [100, 200]

def test_retrieve_subdata_by_query(mock_db):
    # Arrange
    query = "SELECT * FROM test_table"
    columns = ['id', 'name', 'value']
    mock_db.cur.fetchall.return_value = [(1, 'test1', 100), (2, 'test2', 200)]

    # Act
    result_data, result_metadata = mock_db.retrieve_subdata_by_query(query, columns)

    # Assert
    assert result_data.shape == (2, 3)  # Check the shape of the DataFrame
    assert list(result_data.columns) == columns  # Check the columns
    assert result_metadata == [{ 'id': 1 }, { 'id': 2 }]  # Check the metadata

def test_close_connection(mock_db):
    # Act
    mock_db.close_connection()

    # Assert
    mock_db.cur.close.assert_called_once()
    mock_db.connector.close.assert_called_once()