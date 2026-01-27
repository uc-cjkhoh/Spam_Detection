import os
import mlflow
import numpy as np
import pandas as pd

from typing import Any

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
    def save(self, input_sample: np.ndarray):
        """Save / Log Model

        Args:
            input_sample (np.ndarray): input sample saved to mlflow (for reference only)
        """
        
        with mlflow.start_run(run_name='Build/Update Model'):
            mlflow.log_param('model_parameters', self.model.get_params())
            mlflow.sklearn.log_model(
                sk_model=self.model,
                name=self.model_name,
                registered_model_name=f'{self.model_name}',
                input_example=input_sample
            )   
    
    @task(name='Evaluate Model', cache_policy=NO_CACHE)
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """Evaluation the model performance with basic accuracy metrics

        Args:
            x_test (np.ndarray): independent features
            y_test (np.ndarray): dependent feature
        """
        
        y_pred = self.model.predict(x_test)
        
        labels = np.unique(y_test)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=labels,
            average=None
        ) 
        
        df = pd.DataFrame({
            "model": self.model_name,
            "class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })
       
        output_path = "./evaluation/model_metrics.csv"

        df.to_csv(
            output_path,
            mode="a",
            header=not os.path.exists(output_path),
            index=False
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
        
        # use HDBSCAN to create clusters
        hdb = HDBSCAN(min_cluster_size=5)
        cluster_id = hdb.fit_predict(x)
        counts = np.unique(cluster_id, return_counts=True)
        
        # create sample weight based on the number of data in each cluster
        weights = np.asarray([((len(x)) / (len(counts[0]) * counts[1][id])) for id in cluster_id])
        
        # fit to model
        self.model.fit(x, y, sample_weight=weights)
        
        return self.model
    