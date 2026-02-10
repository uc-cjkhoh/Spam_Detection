import mlflow
import numpy as np
import pandas as pd

from datetime import datetime  
from lightgbm.sklearn import LGBMClassifier 
from mlflow.exceptions import MlflowTracingException, RestException

from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from data_validation.ml_validation.validate_model_training import ModelBoneStructureConfig, \
    EvaluateInput, LGBMConfig, TrainInput, PredictInput, PredictProbaInput


class ModelBoneStructure():
    def __init__(self, model_config: ModelBoneStructureConfig): 
        """Define bone structure for any model

        Args:
            model_config (ModelBoneStructureConfig): refer to ModelBoneStructureConfig in validate_model_training.py
        """
        
        self.model_name = model_config.model_name  
        self.model = model_config.model
        
        
    @task(name='Save Model', cache_policy=NO_CACHE)
    def evaluate(self, evaluate_input: EvaluateInput):
        """Save / Log Model

        Args:
            evaluate_input (EvaluateInput): refer to EvaluateInput in validate_model_training.py
        """
        
        logger = get_run_logger()
        
        try:
            with mlflow.start_run(run_name=f'Model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
                mlflow.log_param('model_parameters', self.model.get_params())
                
                x_df = pd.DataFrame(
                    evaluate_input.x_test,
                    columns=[f"f{i}" for i in range(evaluate_input.x_test.shape[1])]
                )
                
                y_s = pd.Series(evaluate_input.y_test, name="y_test")

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
        
        
class LGBM(ModelBoneStructure):
    def __init__(self, lgbm_config: LGBMConfig):
        """Build LGBMClassifier class that inherit from ModelBoneStructure

        Args:
            lgbm_config (LGBMConfig): refer to LGBMConfig in validate_model_training.py
        """
    
        super().__init__( 
            model_name=lgbm_config.model_name,
            model=LGBMClassifier()
        )
            
     
    @task(name='Train Model', cache_policy=NO_CACHE)
    def fit(self, train_input: TrainInput) -> LGBMClassifier:
        """Train model

        Args:
            train_input (TrainInput): refer to TrainInput in validate_model_training.py

        Returns:
            LGBMClassifier: target model
        """
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Training {type(self.model).__name__}-{self.model_name}')
            self.model.fit(train_input.x, train_input.y)
        
        except (ValueError, TypeError) as e:
            logger.error(f'Training failed due to {e}', exc_info=True)
            raise
        
      
    @task(name='Perform Classification', cache_policy=NO_CACHE)
    def predict(self, predict_input: PredictInput) -> np.ndarray:
        """Perform prediction / classification

        Args:
            predict_input (PredictInput): refer to PredictInput in validate_model_training.py

        Returns:
            np.ndarray: result
        """
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Classify data with {type(self.model).__name__}-{self.model_name}')
            return self.model.predict(predict_input.x)
    
        except (ValueError, TypeError) as e:
            logger.error(f'Classification failed due to {e}', exc_info=True)
            raise
    
    
    @task(name='Get Confidence Score', cache_policy=NO_CACHE)
    def predict_proba(self, predict_proba_input: PredictProbaInput) -> np.ndarray:
        """Retrieve prediction or classification probability

        Args:
            predict_proba_input (PredictProbaInput): refer to PredictProbaInput in validate_model_training.py

        Returns:
            np.ndarray: probability list
        """
        
        logger = get_run_logger()
        
        try:
            logger.info(f'Retrieving confidence score of {type(self.model).__name__}-{self.model_name}')
            return self.model.predict_proba(predict_proba_input.x).max(axis=1)
        
        except (ValueError, TypeError) as e:
            logger.error(f'Failed to calculate prediction probability due to {e}', exc_info=True)
            raise