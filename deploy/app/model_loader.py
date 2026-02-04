import mlflow 


class Model:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = None
        
    
    def load(self):
        if self.model == None:
            self.model = mlflow.sklearn.load_model(self.model_uri)
            
        return self.model