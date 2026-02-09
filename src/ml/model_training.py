import mlflow
import numpy as np
import pandas as pd

from typing import Any
from datetime import datetime 
from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import RandomizedSearchCV
from mlflow.exceptions import MlflowTracingException, RestException


class ModelBoneStructure():
    def __init__(self, model_name: str, model: Any): 
        """Define bone structure for any model

        Args:
            experiment_name (str): mlflow experiment name
            model_name (str): model name
            model (Any): any model
        """
        
        self.model_name = model_name  
        self.model = model
        
        
    @task(name='Save Model', cache_policy=NO_CACHE)
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Save / Log Model

        Args:
            input_sample (np.ndarray): input sample saved to mlflow (for reference only)
        """
        
        logger = get_run_logger()
        
        try:
            with mlflow.start_run(run_name=f'Model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
                mlflow.log_param('model_parameters', self.model.get_params())
                
                x_df = pd.DataFrame(
                    x_test,
                    columns=[f"f{i}" for i in range(x_test.shape[1])]
                )
                
                y_s = pd.Series(y_test, name="y_test")

                signature = mlflow.models.signature.infer_signature(
                    x_df, self.model.predict(x_df)
                )
                
                model_info = mlflow.sklearn.log_model(
                    sk_model=self.model,
                    name=self.model_name,
                    registered_model_name=self.model_name,
                    signature=signature,
                    input_example=x_df.head(5)
                )  
                
                eval_data = x_df.copy()
                eval_data["y_test"] = y_s
    
                mlflow.models.evaluate(
                    model_info.model_uri,
                    data=eval_data,
                    targets='y_test',
                    model_type='classifier'
                )
        
        except MlflowTracingException as e:
            logger.error(f'Failed to trace mlflow server uri: {e}', exc_info=True)
            raise
        except RestException as e:
            logger.error(f'Connection failure: {e}', exc_info=True)
            raise
        except (ValueError, TypeError) as e:
            logger.error(f'Model evaluation / save failed due to {e}', exc_info=True)
            raise
        
        
class SGD(ModelBoneStructure):
    def __init__(self, model_name: str):
        """Build SGDClassifier class that inherit from ModelBoneStructure

        Args:
            experiment_name (str): mlflow experiment name
            model_name (str): model name
        """
    
        super().__init__( 
            model_name=model_name,
            model=LGBMClassifier()
        )
            
     
    @task(name='Train Model', cache_policy=NO_CACHE)
    def fit(self, x: np.ndarray, y: np.ndarray) -> LGBMClassifier:
        """Train model

        Args:
            x (np.ndarray): independent features
            y (np.ndarray): dependent feature

        Returns:
            LGBMClassifier: target model
        """
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Training {type(self.model).__name__}-{self.model_name}')
            self.model.fit(x, y)
        
        except (ValueError, TypeError) as e:
            logger.error(f'Training failed due to {e}', exc_info=True)
            raise
        
      
    @task(name='Perform Classification', cache_policy=NO_CACHE)
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform prediction / classification

        Args:
            x (np.ndarray): independent features

        Returns:
            np.ndarray: result
        """
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Classify data with {type(self.model).__name__}-{self.model_name}')
            return self.model.predict(x)
    
        except (ValueError, TypeError) as e:
            logger.error(f'Classification failed due to {e}', exc_info=True)
            raise
    
    
    @task(name='Get Confidence Score', cache_policy=NO_CACHE)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Retrieve prediction or classification probability

        Args:
            x (np.ndarray): independent features

        Returns:
            np.ndarray: probability list
        """
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Retrieving confidence score of {type(self.model).__name__}-{self.model_name}')
            return self.model.predict_proba(x).max(axis=1)
        
        except (ValueError, TypeError) as e:
            logger.error(f'Failed to calculate prediction probability due to {e}', exc_info=True)
            raise