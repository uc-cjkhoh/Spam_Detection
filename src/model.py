import pandas as pd
import xgboost
import yaml 
import os

from tqdm import tqdm
from addict import Dict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import HDBSCAN, SpectralClustering
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSequenceClassification

from . decorators import timer, error_log

_config_path = r'./configs/config.yaml'
if not os.path.exists(_config_path):
    raise FileNotFoundError(f'Config file is not found in {_config_path}')
with open(_config_path) as f:
    cfg = Dict(yaml.load(f, Loader=yaml.FullLoader))


@error_log
@timer
def text_embedding(messages: pd.Series):
    model = SentenceTransformer(cfg.models.text_embedding.model_name, trust_remote_code=True)
    embeddings = model.encode(
        messages,  
        batch_size=cfg.models.text_embedding.batch_size, 
        show_progress_bar=True
    )
    return embeddings 
    

@error_log
@timer
def train_model(data: pd.Series) -> pd.DataFrame: 
    def text_pipe(texts: pd.DataFrame, batch_size=32):
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            results.extend(pipe(batch))
        return results
    
    if cfg.models.spam_detection.model_type != 'text-classification':
        raise ValueError(f'Invalid model type: `{cfg.models.spam_detection.model_type}`, check your config.yaml')
    
    pipe = pipeline(cfg.models.spam_detection.model_type, model=cfg.models.spam_detection.model_name)
    
    prediction = text_pipe(data.to_list()) 
    label = [p['label'] for p in prediction]
    score = [p['score'] for p in prediction]
    
    return pd.DataFrame({
        'Message': data,
        'Label': label,
        'Score': score
    })