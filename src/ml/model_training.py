import mlflow
import numpy as np
import pandas as pd

from datetime import datetime  
from lightgbm.sklearn import LGBMClassifier 
from mlflow.exceptions import MlflowTracingException, RestException
from sklearn.metrics import accuracy_score, classification_report

from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from data_validation.ml_validation.validate_model_training import ModelBoneStructureConfig


class ModelBoneStructure():
    def __init__(self, model_config: ModelBoneStructureConfig): 
        """Define bone structure for any model

        Args:
            model_config (ModelBoneStructureConfig): refer to ModelBoneStructureConfig in validate_model_training.py
        """
        
        self.model_name = model_config.model_name  
        self.model = model_config.model
        
        
    @task(name='Save Model', cache_policy=NO_CACHE)
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Save / Log Model

        Args:
            x_test (np.ndarray): independent input
            y_test (np.ndarray): dependent input
        """
        
        if not isinstance(x_test, np.ndarray):
            raise TypeError('x_test need to be in numpy ndarray type')
        elif not isinstance(y_test, np.ndarray):
            raise TypeError('y_test need to be in numpy ndarray type')
        
        logger = get_run_logger()
        
        try:
            with mlflow.start_run(run_name=f'Model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
                mlflow.log_param('model_parameters', self.model.get_params())
                
                x_df = pd.DataFrame(
                    x_test,
                    columns=[f"f{i}" for i in range(x_test.shape[1])]
                )
                 
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
                eval_data["y_test"] = np.array(y_test).flatten()
                
                # Check if model can predict all classes
                y_pred_proba = self.model.predict_proba(x_df)
                if y_pred_proba.shape[1] != len(np.unique(y_test)):
                    logger.warning("Model predict_proba does not have probabilities for all classes in y_test, skipping MLflow evaluation")
                    return
    
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
        
        
class LGBM(ModelBoneStructure):
    def __init__(self, model_name: str):
        """Build LGBMClassifier class that inherit from ModelBoneStructure

        Args:
            model_name (str): model name
        """
        
        if not isinstance(model_name, str):
            raise TypeError('Model name need to be string type')
    
        super().__init__( 
            model_config=ModelBoneStructureConfig( 
                model_name=model_name,
                model=LGBMClassifier(class_weight={0: 10, 1: 1})
            )
        )
            
     
    @task(name='Train Model', cache_policy=NO_CACHE)
    def fit(self, x: np.ndarray, y: np.ndarray) -> LGBMClassifier:
        """Train model

        Args:
            x (np.ndarray): independent input
            y (np.ndarray): dependent input
            
        Returns:
            LGBMClassifier: target model
        """
        
        if not isinstance(x, np.ndarray):
            raise TypeError('x need to be in numpy ndarray type')
        elif not isinstance(y, np.ndarray):
            raise TypeError('y need to be in numpy ndarray type')
        
        
        logger = get_run_logger()
         
        logger.info(f'Training {type(self.model).__name__}-{self.model_name}')
        self.model.fit(x, y) 
      
      
    @task(name='Perform Classification', cache_policy=NO_CACHE)
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform prediction / classification

        Args:
            x (np.ndarray): independent inputs

        Returns:
            np.ndarray: result
        """
        
        if not isinstance(x, np.ndarray):
            raise TypeError('x need to be in numpy ndarray type') 
        
        logger = get_run_logger()
         
        logger.info(f'Classify data with {type(self.model).__name__}-{self.model_name}')
        return self.model.predict(x)
     
    
    @task(name='Get Confidence Score', cache_policy=NO_CACHE)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Retrieve prediction or classification probability

        Args:
            x (np.ndarray): independent inputs

        Returns:
            np.ndarray: probability list
        """
        
        if not isinstance(x, np.ndarray):
            raise TypeError('x need to be in numpy ndarray type') 
         
        logger = get_run_logger()
        
        try:
            logger.info(f'Retrieving confidence score of {type(self.model).__name__}-{self.model_name}')
            return self.model.predict_proba(x).max(axis=1)
        
        except (ValueError, TypeError) as e:
            logger.error(f'Failed to calculate prediction probability due to {e}', exc_info=True)
            raise