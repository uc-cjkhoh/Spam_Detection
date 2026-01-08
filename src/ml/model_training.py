import mlflow

from abc import ABC
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier


class ModelBoneStructure(ABC):
    def __init__(self, experiment_name, model_name, model): 
        self.model_name = model_name  
        
        self.experiment = mlflow.search_experiments(
            filter_string=f"name='{experiment_name}'", 
            order_by=['creation_time DESC']
        )[0].experiment_id
        
        self.model_list = mlflow.search_logged_models(
            experiment_ids=[self.experiment], 
            filter_string=f"name='{model_name}'", 
            order_by=[{'field_name': 'creation_timestamp', 'ascending': False}]
        )
        
        self.model = model  
         
    def predict(self, x):
        return self.model.predict(x) 
    
    def predict_proba(self, x):
        return self.model.predict_proba(x).max(axis=1)
    
    def get_existing_models(self):
        return self.model_list

class SGD(ModelBoneStructure):
    def __init__(self, experiment_name):
        super().__init__(
            experiment_name=experiment_name,
            model_name=type(SGDClassifier).__name__,
            model=SGDClassifier(
                loss='log_loss', 
                class_weight='balanced',
                early_stopping=True,
                learning_rate='adaptive',
                validation_fraction=0.2
            )
        )
       
class XGBoost(ModelBoneStructure):
    def __init__(self, experiment_name):
        super().__init__(
            experiment_name=experiment_name, 
            model_name=type(XGBClassifier).__name__,
            model=XGBClassifier(
                n_estimators=500,
                base_score=0.5
            )
        )
        
