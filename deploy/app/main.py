import mlflow

from typing import Union, List
from fastapi import FastAPI 
from app.model_loader import Model
from app.config.settings import Settings


mlflow.set_tracking_uri('http://10.168.49.12:5000')

app = FastAPI() 
settings = Settings()
model = Model(settings.model_uri)
      
@app.post('/classify')
def classify(payload: Union[str, List[str]]):
    # transform data
    
    # perform classification
    
    
    pass 
 