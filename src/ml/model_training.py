import mlflow

from abc import ABC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from prefect import task
from prefect.cache_policies import NO_CACHE


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
        
        self.model = model if len(self.model_list) == 0 else mlflow.sklearn.load_model(f'models:/{model_name}/{len(self.model_list)}')
         
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
                validation_fraction=0.2,
                eta0=0.01
            )
        )
    
    @task(name='Fit Model', cache_policy=NO_CACHE)
    def fit(self, x, y):
        self.model.fit(x, y)
        return self.model
    
    
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
        
    @task(name='Fit Model', cache_policy=NO_CACHE)
    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)  
        self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)], xgb_model=self.model.get_booster()) 
        return self.model