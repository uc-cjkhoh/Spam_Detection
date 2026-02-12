import joblib 
import mlflow
import numpy as np
import pandas as pd

from fastapi import FastAPI, BackgroundTasks
from langchain_huggingface import HuggingFaceEmbeddings 

from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
from sqlalchemy.exc import StatementError, CompileError

from app.schema import InputData, DataToMySQL
from app.config.config_loader import get_config
from app.preprocessing import feature_engineering


# ================== setup environment ==================
app = FastAPI()

config = get_config()

mlflow.set_tracking_uri(config['mlflow_uri'])

model = mlflow.sklearn.load_model(config['model_uri'])
 
engine = create_engine(
    URL.create(
        drivername='mysql+pymysql',
        host=config['database']['host'],
        port=config['database']['port'],
        username=config['database']['user'],
        password=config['database']['password'],
        database=config['database']['schema']
    )
)

embedding_model = HuggingFaceEmbeddings(
    model_name=config['embedding']['model_name'],  
    encode_kwargs={
        'batch_size': config['embedding']['batch_size'], 
        'normalize_embeddings': config['embedding']['normalize_embeddings']
    }
)

scaler = joblib.load(f'./app/scaler/{config["experiment_name"]}_standard_scaler.joblib')

 
# ================== background tasks ==================
def save_to_mysql(params: DataToMySQL):
    """Save classified result to mysql

    Args:
        params (DataToMySQL): pydantic, check more on scheme.py
    """
    
    try: 
        data = pd.DataFrame({
            'id': params.id,
            'spam_label': params.result,
            'confidence_score': params.confidence_score
        })
        
        with engine.begin() as conn:
            for row in data.to_dict(orient='records'):
                conn.execute(
                    text("""
                        INSERT IGNORE INTO api_result_2 (id, spam_label, confidence_score)
                        VALUES (:id, :spam_label, :confidence_score)
                    """),
                    row
                )
 
    except StatementError as e:
        print(f'Failed to upload data due to {e}') 
    except CompileError as e:
        print(f'Failed to upload data due to {e}') 
        

# ================== define api ==================
@app.post('/classify')
async def classify(item: InputData, background_tasks: BackgroundTasks):
    """Perform classification and return data in json format

    Args:
        item (InputData): request data, check more on scheme.py

    Returns:
        json: classification result and confidence score
    """
    
    sms = pd.Series(item.payload)
    
    features = feature_engineering(sms) 
    features = scaler.transform(features)
    
    embedding = embedding_model.embed_documents(item.payload)
    
    n, features_dim = features.shape
    e_dim = embedding.shape[1]
    
    x = np.empty(n, features_dim + e_dim)
    x[:, :features_dim] = features
    x[:, features_dim:] = embedding
    
    result = model.predict(x).tolist()
    
    confidence_score = model.predict_proba(x).max(axis=1).tolist()

    background_tasks.add_task(save_to_mysql, DataToMySQL(id=item.id, result=result, confidence_score=confidence_score))

    return {'result': result, 'confidence_score': confidence_score}
 
 
