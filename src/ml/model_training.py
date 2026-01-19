import mlflow
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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
    def __init__(self, experiment_name):
        super().__init__(
            experiment_name=experiment_name,
            model_name=SGDClassifier.__name__,
            model=SGDClassifier(
                loss='log_loss', 
                class_weight='balanced',
                early_stopping=True,
                learning_rate='adaptive',
                validation_fraction=0.2,
                eta0=0.01
            )
        )
     
    def fit(self, x, y, skip_initialization):
        try:
            check_is_fitted(self.model)
            if not skip_initialization:
                raise NotFittedError('Not Skip Initialization')
            
            smote = SMOTE(random_state=42)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            
            classes = [0, 1]
            best_loss = 0
            epochs_no_prove = 0
            patience = 5
            epochs = 300
            
            for i in range(epochs):
                batch_size = len(x_train) // 10
                idx = np.random.permutation(len(x_train))[:batch_size]
                batch_x = x_train[idx]
                batch_y = y_train[idx]
            
                balanced_x, balanced_y = smote.fit_resample(batch_x, batch_y)
                balanced_x, balanced_y = balanced_x.astype(np.float32), balanced_y.astype(np.float32)
            
                # training
                self.model.partial_fit(balanced_x, balanced_y, classes=classes)
                  
                # batch evaluation
                y_pred = self.predict_proba(x_test)
                loss = log_loss(y_test, y_pred)
                print(f'Epoch {i} - Log Loss: {loss:.5f}')
                
                if (best_loss == 0) | (loss < best_loss):
                    best_loss = loss
                else:
                    epochs_no_prove += 1
            
                if epochs_no_prove >= patience:
                    break
 
        except NotFittedError as e:
            self.model.fit(x, y)
            
        return self.model
    
class XGBoost(ModelBoneStructure):
    def __init__(self, experiment_name):
        super().__init__(
            experiment_name=experiment_name, 
            model_name=XGBClassifier.__name__,
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