import pandas as pd 

from sqlalchemy import create_engine, text, MetaData, Table, Column, BigInteger, DateTime, SmallInteger, Boolean, Double, String
from sqlalchemy.dialects.mysql import insert

from prefect import task
from prefect.cache_policies import NO_CACHE 


class Database:
    def __init__(self, host: str, port: int, user: str, password: str):
        """
        Initiate MySQL Connection

        Args:
            host (str): host name
            port (int): port number
            user (str): username
            password (str): password
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.engine = create_engine("mysql+pymysql://unified:unified@10.168.51.196:3306/sms_spam_cd")
  
  
    @task(name="Run SQL Statement")
    def run_statement(self, statement: str):
        """
        Run SQL statement like DDL, DML

        Args:
            statement (str): statement to run
        """
        with self.engine.begin() as conn:
            conn.execute(text(statement))
      
    
    @task(name="Retrieve Records From MySQL", cache_policy=NO_CACHE)  
    def get_records(self, query: str) -> pd.DataFrame:
        """
        Retrieve data from MySQL

        Args:
            query (str): query to run

        Returns:
            pd.DataFrame
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
            
        return pd.DataFrame(rows, columns=columns)
    
    
    @task(name="Save to MySQL", cache_policy=NO_CACHE)
    def save_to_mysql(self, data: dict):
        """
        Save data to MySQL

        Args:
            data (dict): data to save
        """ 
        metadata = MetaData()
        target_table = Table(
            "metadata_result",
            metadata,
            Column('row_id', BigInteger, primary_key=True),
            Column('id', BigInteger, nullable=False),
            Column('datetime', DateTime, nullable=True),
            Column('spam_label', SmallInteger, nullable=True),
            Column('confidence_score', Double, nullable=True),
            Column('label_status', String(20), nullable=True),
            Column('model', String(20), nullable=True),
            Column('last_batch', Boolean, nullable=True),
            schema='sms_spam_cd'
        )
        
        metadata.create_all(self.engine)
        
        with self.engine.connect() as conn:
            insert_statement = insert(target_table).values(data)
        
            on_duplicate_key_statement = insert_statement.on_duplicate_key_update(
                spam_label=insert_statement.inserted.spam_label,
                confidence_score=insert_statement.inserted.confidence_score,
                label_status=insert_statement.inserted.label_status,
                last_batch=insert_statement.inserted.last_batch,
                model=insert_statement.inserted.model
            )
            
            conn.execute(on_duplicate_key_statement)
            conn.commit() 
            
    
    @task(name="Disconnect MySQL", cache_policy=NO_CACHE)
    def close_connection(self):
        """
        Disconnect connected MySQL
        """
        self.engine.dispose()