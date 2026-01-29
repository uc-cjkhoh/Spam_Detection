import os
import mlflow
import numpy as np
import pandas as pd

from typing import Any
from datetime import datetime

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import HDBSCAN 

from prefect import task
from prefect.cache_policies import NO_CACHE


class ModelBoneStructure():
    def __init__(self, experiment_name: str, model_name: str, model: Any): 
        """Define bone structure for any model

        Args:
            experiment_name (str): mlflow experiment name
            model_name (str): model name
            model (Any): any model
        """
        
        self.model_name = model_name  
        
        self.experiment = mlflow.search_experiments(
            filter_string=f"name='{experiment_name}'"
        )[0].experiment_id
        
        self.model_list = mlflow.search_logged_models(
            experiment_ids=[self.experiment], 
            filter_string=f"name='{model_name}'", 
            order_by=[{'field_name': 'creation_timestamp', 'ascending': False}]
        )
        
        self.model = model if len(self.model_list) == 0 else mlflow.sklearn.load_model(self.model_list.iloc[0].artifact_location)
    
    
    @task(name='Perform Classification', cache_policy=NO_CACHE)
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform prediction / classification

        Args:
            x (np.ndarray): independent features

        Returns:
            np.ndarray: result
        """
        
        return self.model.predict(x)
    
    
    @task(name='Get Confidence Score', cache_policy=NO_CACHE)
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Retrieve prediction or classification probability

        Args:
            x (np.ndarray): independent features

        Returns:
            np.ndarray: probability list
        """
        
        return self.model.predict_proba(x).max(axis=1)


    @task(name='Save Model', cache_policy=NO_CACHE)
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Save / Log Model

        Args:
            input_sample (np.ndarray): input sample saved to mlflow (for reference only)
        """
        
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
                name=f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
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
       
      
class SGD(ModelBoneStructure):
    @task(name="Create New SGDClassifier Object", cache_policy=NO_CACHE)
    def __init__(self, experiment_name: str, model_name: str):
        """Build SGDClassifier class that inherit from ModelBoneStructure

        Args:
            experiment_name (str): mlflow experiment name
            model_name (str): model name
        """
        
        super().__init__(
            experiment_name=experiment_name,
            model_name=model_name,
            model=SGDClassifier(
                loss='log_loss', 
                class_weight='balanced'
            )
        )
     
     
    @task(name='Train Model', cache_policy=NO_CACHE)
    def fit(self, x: np.ndarray, y: np.ndarray) -> SGDClassifier:
        """Train model

        Args:
            x (np.ndarray): independent features
            y (np.ndarray): dependent feature

        Returns:
            SGDClassifier: target model
        """
        
        # # use HDBSCAN to create clusters
        # hdb = HDBSCAN(min_cluster_size=5)
        # cluster_id = hdb.fit_predict(x)
        # counts = np.unique(cluster_id, return_counts=True)
        
        # # create sample weight based on the number of data in each cluster
        # weights = np.asarray([((len(x)) / (len(counts[0]) * counts[1][id])) for id in cluster_id])
        
        # fit to model
        self.model.fit(x, y)
        
        return self.model
    