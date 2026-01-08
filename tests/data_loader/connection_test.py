import pandas as pd
import pytest  
from unittest.mock import MagicMock, patch 

from src.data_loader.connection import Database
  
        
@pytest.fixture
def mock_mysql():
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(1, "Alice"), (2, "Bob")]

    mock_connector = MagicMock()
    mock_connector.cursor.return_value = mock_cursor

    with patch("mysql.connector.connect", return_value=mock_connector):
        yield mock_connector, mock_cursor


def test_initialize_db_connection(mock_mysql):
    mock_connector, _ = mock_mysql

    db = Database("localhost", 3306, "user", "pw")
    
    assert db.connector is mock_connector
    assert db.cur is mock_connector.cursor.return_value


def test_run_query(mock_mysql):
    mock_connector, mock_cursor = mock_mysql

    db = Database("host", 3306, "user", "pw")
    result = db.run_query(query="SELECT * FROM tbl", columns=["id", "name"])

    mock_cursor.execute.assert_called_once_with("SELECT * FROM tbl")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["id", "name"]
    assert result.shape == (2, 2)
 

def test_close_connection(mock_mysql):
    mock_connector, mock_cursor = mock_mysql
    db = Database("h", 1, "u", "p")

    db.close_connection()

    mock_cursor.close.assert_called_once()
    mock_connector.close.assert_called_once()
