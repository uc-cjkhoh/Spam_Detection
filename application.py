from fastapi import FastAPI
from pydantic import BaseModel

import os
import joblib
import pandas as pd

from loader.config_loader import cfg
from src.model import text_embedding, _predictive_model

model_folder = cfg.models.save_model_to.folder
model_filename = f'{type(_predictive_model).__name__}.joblib'
filepath = os.path.join(model_folder, model_filename)

model = joblib.load(filepath)
app = FastAPI(title='Spam Detection Module')

class ModelData(BaseModel):
    message: str

@app.post('/predict')
def predict(data: ModelData):
    message = text_embedding(pd.Series(data.message))
    prediction = model.predict(message)
    confidence_score = model.predict_proba(message)
    return {
        'message': data.message,
        'is_spam': bool(prediction[0]),
        'confidence': confidence_score.max(axis=1)[0]
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)