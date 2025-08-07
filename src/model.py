import pandas as pd 

from tqdm import tqdm 
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSequenceClassification
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report 

from loader.config_loader import cfg
from . decorators import timer, error_log 
  
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
    def text_pipe(texts: pd.DataFrame, batch_size=16):
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            results.extend(pipe(batch))
        return results
     
    pipe = pipeline('text-classification', model=cfg.models.spam_detection.model_name)
    
    prediction = text_pipe(data.to_list(), cfg.models.spam_detection.batch_size) 
    label = LabelEncoder()
    label = label.fit_transform([p['label'] for p in prediction])
    score = [p['score'] for p in prediction]
    
    return pd.DataFrame({
        cfg.data.target_column: data,
        cfg.data.target_column + '_label': label,
        cfg.data.target_column + '_score': score
    })