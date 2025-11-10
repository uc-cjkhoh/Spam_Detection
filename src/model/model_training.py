import pickle
import numpy as np
import mlflow
import mlflow.sklearn
   
   
class MLPipeline:
    def __init__(self, model):
        self.model = model
        
    def classify_message(self, model_id, embeddings, metadata=None):  
        dataset = mlflow.data.from_numpy(embeddings, source=str(metadata))
        
        with mlflow.start_run():
            mlflow.log_param('model', str(type(self.model).__name__))
            mlflow.log_input(dataset, context="training")

            pred = self.model.predict(embeddings)

    def get_model(self):
        return self.model