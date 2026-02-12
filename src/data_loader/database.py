import pandas as pd 

from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ArgumentError, OperationalError, StatementError, CompileError
  
from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

 
class Database:
    def __init__(self, host: str, port: int, username: str, password: str, table_schema: str):
        """
        Initiate MySQL Connection

        Args:
            host (str): host name 
            port (int): port number 
            username (str): username
            password (str): password
            table_schema (str): schema
        """
        
        logger = get_run_logger()
         
        self.schema = table_schema
        
        try:
            connection_url = URL.create(
                drivername='mysql+pymysql',
                host=host,
                port=port,
                username=username,
                password=password,
                database=table_schema
            )
            
            logger.info(f'Connecting to: {connection_url.render_as_string(hide_password=True)}')
            self.engine = create_engine(connection_url)

        except ArgumentError:
            logger.error('Invalid SQLAlchemy configuration', exc_info=True)
            raise
        except OperationalError:
            logger.error('Database connection error', exc_info=True)
            raise
        except (ValueError, TypeError) as e:
            logger.error(f'Failed to connect to MySQL due to {e}', exc_info=True)
            raise
  
  
    @task(name="Run SQL Statement", cache_policy=NO_CACHE)
    def run_statement(self, statement: str):
        """
        Run SQL statement like DDL, DML

        Args:
            statement (str): statement to run
        """
        
        if not isinstance(statement, str):
            raise TypeError('Statement must be in string type')
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Executing statement: {statement}', exc_info=True)
            with self.engine.begin() as conn:
                conn.execute(text(statement))
        
        except StatementError as e:
            logger.error(f'Invalid statement: {e}', exc_info=True)
            raise
        except CompileError as e:
            logger.error(f'Compile error when trying: {e}', exc_info=True)
            raise
      
    
    @task(name="Retrieve Records From MySQL", cache_policy=NO_CACHE)  
    def get_records(self, query: str) -> pd.DataFrame:
        """
        Retrieve data from MySQL

        Args:
            query (str): query to run

        Returns:
            pd.DataFrame
        """
        
        if not isinstance(query, str):
            raise TypeError('Query must be in string type')
        
        logger = get_run_logger()
        
        try: 
            with self.engine.connect() as conn:
                data = pd.read_sql(text(query), conn) 
                
            return data

        except StatementError as e:
            logger.error(f'Invalid query: {e}', exc_info=True)
            raise
        except CompileError as e:
            logger.error(f'Compile error when trying: {e}', exc_info=True)
            raise
        
    
    @task(name="Save to MySQL", cache_policy=NO_CACHE)
    def save_to_mysql(self, data: pd.DataFrame, destination_table: str, on_duplicate: list[str] = None):
        """
        Save data to MySQL

        Args:
            data (dict): data to save
            destination_table (str): target table 
            on_duplicate (list[str]): duplicate update criteria
        """ 
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data need to be in list type')
        
        logger = get_run_logger()
        
        columns_list = ', '.join(data.columns.to_list())
        values_list = ':' + ', :'.join(data.columns.to_list())
        on_duplicate_list = 'ON DUPLICATE KEY UPDATE' + ' ' + ', '.join(['{} = VALUES({})'.format(item, item) for item in on_duplicate]) \
            if on_duplicate is not None else ''
        
        try:  
            insert_sql = text(
                f"""
                    INSERT INTO {self.schema}.{destination_table} ({columns_list})
                    VALUES ({values_list})
                    {on_duplicate_list}
                """
            )

            with self.engine.connect() as conn:
                try: 
                    conn.execute(insert_sql, data.to_dict(orient="records"))   
                    conn.commit()

                except OperationalError as e:
                    conn.rollback()
                    logger.error(f'Database connection issue: {e}', exc_info=True)
                    raise
                
        except ArgumentError as e:
            logger.error(f'Invalid argument: {e}', exc_info=True)
            raise
        except OperationalError as e:
            logger.error(f'Database unreachable or timeout: {e}', exc_info=True)
            raise 
        except (ValueError, TypeError) as e:
            logger.error(f'Failed to save data due to {e}', exc_info=True)
            raise
        
        
    @task(name="Disconnect MySQL", cache_policy=NO_CACHE)
    def close_connection(self):
        """Disconnect connected MySQL"""
        self.engine.dispose()
        