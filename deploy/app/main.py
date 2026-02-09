import numpy as np
import pandas as pd
import mlflow
mlflow.set_tracking_uri('http://10.168.49.12:5000')

from fastapi import FastAPI 
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings 
from app.config.config_loader import get_config
from app.preprocessing import feature_engineering

 
app = FastAPI()

config = get_config()

model = mlflow.sklearn.load_model(config['model_uri'])

embedding_model = HuggingFaceEmbeddings(
    model_name=config['embedding']['model_name'], 
    model_kwargs={'trust_remote_code': True}, 
    encode_kwargs={
        'batch_size': config['embedding']['batch_size'], 
        'normalize_embeddings': config['embedding']['normalize_embeddings']
    }, 
    show_progress=config['embedding']['show_progress']
)


class InputData(BaseModel):
    payload: list[str]
    

@app.post('/classify')
async def classify(item: InputData):
    sms = pd.Series(item.payload)
    features = feature_engineering(sms)
    embedding = embedding_model.embed_documents(item.payload)
    x = np.hstack((features, embedding))
    
    return {'result': model.predict(x).tolist()}
 