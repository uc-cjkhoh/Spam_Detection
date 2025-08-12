from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd

from src.model import text_embedding

model = joblib.load('models/SGDClassifier-20250811.joblib')
app = FastAPI(title='Spam Detection Model (SVC)')

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