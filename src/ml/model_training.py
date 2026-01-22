import mlflow

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError 
from xgboost import XGBClassifier 
from sklearn.cluster import HDBSCAN
from collections import Counter


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
        
    def predict(self, x):
        return self.model.predict(x)
    
    def predict_proba(self, x):
        return self.model.predict_proba(x).max(axis=1)
    
    def get_existing_models(self):
        return self.model_list
  
class SGD(ModelBoneStructure):
    def __init__(self, experiment_name, model_name):
        super().__init__(
            experiment_name=experiment_name,
            model_name=model_name,
            model=SGDClassifier(
                loss='log_loss', 
                class_weight='balanced'
            )
        )
     
    def fit(self, x, y):
        # hdb = HDBSCAN(min_cluster_size=3)
        # cluster_id = hdb.fit_predict(x)
        
        # counts = Counter(cluster_id)
        # counts.pop(-1, None)
          
        self.model.fit(x, y)
        return self.model
    
class XGBoost(ModelBoneStructure):
    def __init__(self, experiment_name, model_name):
        super().__init__(
            experiment_name=experiment_name, 
            model_name=model_name,
            model=XGBClassifier(
                n_estimators=500,
                base_score=0.5
            )
        )
         
    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        
        try:
            check_is_fitted(self.model)
            self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)], xgb_model=self.model.get_booster()) 
        except NotFittedError as e: 
            self.model.fit(x_train, y_train, eval_set=[(x_test, y_test)]) 
            
        return self.model