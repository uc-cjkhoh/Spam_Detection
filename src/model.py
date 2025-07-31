import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost
import yaml
import os

from addict import Dict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import HDBSCAN, SpectralClustering
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sentence_transformers import SentenceTransformer

from . decorators import timer

_config_path = r'./configs/config.yaml'
if not os.path.exists(_config_path):
    raise FileNotFoundError(f'Config file is not found in {_config_path}')
with open(_config_path) as f:
    cfg = Dict(yaml.load(f, Loader=yaml.FullLoader))


@timer
def text_embedding(messages: pd.Series):
    model = SentenceTransformer(cfg.models.text_embedding.model_name, trust_remote_code=True)

    embeddings = model.encode(
        messages,  
        batch_size=cfg.models.text_embedding.batch_size, 
        show_progress_bar=True
    )

    # similarities = model.similarity(embeddings, embeddings)
    
    return embeddings 
    
    
@timer
def train_model(data: dict):
    _models = {
        'lr': LogisticRegression(),
        'spec_clst': SpectralClustering(),
        'hdbscan': HDBSCAN(),
        'svc': SVC(),
        'gbc': GradientBoostingClassifier(),
        'xgb': xgboost
    }
    
    scale_types = data.keys()
    for i, scale_type in enumerate(scale_types):
        temp_data = data[scale_type]
        
