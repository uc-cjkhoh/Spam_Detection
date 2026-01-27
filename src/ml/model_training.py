import os
import mlflow
import numpy as np
import pandas as pd
 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import HDBSCAN 

from prefect import task
from prefect.cache_policies import NO_CACHE


class ModelBoneStructure():
    def __init__(self, experiment_name, model_name, model): 
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
    def predict(self, x):
        return self.model.predict(x)
    
    @task(name='Get Confidence Score', cache_policy=NO_CACHE)
    def predict_proba(self, x):
        return self.model.predict_proba(x).max(axis=1)

    @task(name='Save Model', cache_policy=NO_CACHE)
    def save(self, input_sample):
        with mlflow.start_run(run_name='Build/Update Model'):
            mlflow.log_param('model_parameters', self.model.get_params())
            mlflow.sklearn.log_model(
                sk_model=self.model,
                name=self.model_name,
                registered_model_name=f'{self.model_name}',
                input_example=input_sample
            )   
    
    @task(name='Evaluate Model', cache_policy=NO_CACHE)
    def evaluate(self, x_test, y_test):
        """Evaluation the model performance with basic accuracy metrics"""
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
       
        output_path = "./logs/evaluation/model_metrics.csv"

        df.to_csv(
            output_path,
            mode="a",
            header=not os.path.exists(output_path),
            index=False
        )
                
      
class SGD(ModelBoneStructure):
    @task(name="Create New SGDClassifier Object", cache_policy=NO_CACHE)
    def __init__(self, experiment_name, model_name):
        super().__init__(
            experiment_name=experiment_name,
            model_name=model_name,
            model=SGDClassifier(
                loss='log_loss', 
                class_weight='balanced'
            )
        )
     
    @task(name='Train Model', cache_policy=NO_CACHE)
    def fit(self, x, y):
        hdb = HDBSCAN(min_cluster_size=5)
        cluster_id = hdb.fit_predict(x)
        counts = np.unique(cluster_id, return_counts=True)
        weights = np.asarray([((len(x)) / (len(counts[0]) * counts[1][id])) for id in cluster_id])
        self.model.fit(x, y, sample_weight=weights)
        return self.model
    