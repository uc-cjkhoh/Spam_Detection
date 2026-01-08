import pandas as pd
import mysql.connector 

from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, DateTime, SmallInteger, Boolean, Double, String
from sqlalchemy.dialects.mysql import insert

  
class Database:
    def __init__(self, host, port, user, password):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connector = self.initialize_db_connection()
        self.cur = self.connector.cursor()
 
    def initialize_db_connection(self):
        try:     
            return mysql.connector.connect(
                host = self.host,
                port = self.port,
                user = self.user,
                password = self.password
            )
        except Exception as e:
            raise Exception(e)
      
    def run_query(self, query, columns: list) -> pd.DataFrame:
        self.cur.execute(query)
        data = pd.DataFrame(self.cur.fetchall(), columns=columns)
        return data
    
    def close_connection(self):
        try:
            if self.cur:
                self.cur.close()
        except Exception as e:
            raise Exception(f"Warning: Failed to close cursor: {e}")
        
        try:
            if self.connector:
                self.connector.close()
        except Exception as e:
            raise Exception(f"Warning: Failed to close connector: {e}")
        
     
    def save_to_mysql(self, data: dict):
        engine = create_engine(f'mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/sms_spam_cd')
        metadata = MetaData()
        target_table = Table(
            "metadata_result",
            metadata,
            Column('row_id', BigInteger, primary_key=True),
            Column('id', BigInteger, nullable=False),
            Column('datetime', DateTime, nullable=False),
            Column('spam_label', SmallInteger, nullable=True),
            Column('confidence_score', Double, nullable=True),
            Column('label_status', String(20), nullable=True),
            Column('model', String(20), nullable=False),
            schema='sms_spam_cd'
        )
        
        metadata.create_all(engine)
        
        with engine.connect() as conn:
            insert_statement = insert(target_table).values(data)
        
            on_duplicate_key_statement = insert_statement.on_duplicate_key_update(
                spam_label=insert_statement.inserted.spam_label,
                confidence_score=insert_statement.inserted.confidence_score,
                label_status=insert_statement.inserted.label_status
            )
            
            conn.execute(on_duplicate_key_statement)
            conn.commit()
            
        engine.dispose() 