import pandas as pd 
import numpy as np 

from tqdm import tqdm  
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer 

from loader.config_loader import cfg 
from .decorators import timer, error_log 
  

_model = SentenceTransformer(cfg.models.text_embedding.model_name, trust_remote_code=True)
_tokenizer = AutoTokenizer.from_pretrained(cfg.models.text_embedding.model_name, trust_remote_code=True)

_pipe = pipeline('text-classification', model=cfg.models.spam_detection.model_name)
_tokenizer = AutoTokenizer.from_pretrained(cfg.models.spam_detection.model_name)


@error_log
@timer
def text_embedding(messages: pd.Series) -> np.ndarray:
    messages = messages.fillna("").astype(str)
    
    # Tokenize with truncation
    truncated_texts = []
    for msg in messages.to_list():
        tokens = _tokenizer(
            msg,
            max_length=8192,
            truncation=True
        )
        # Decode back to text (SentenceTransformer still needs text input)
        truncated_texts.append(_tokenizer.decode(tokens["input_ids"], skip_special_tokens=True))
    
    return _model.encode(
        truncated_texts,  
        batch_size=cfg.models.text_embedding.batch_size, 
        show_progress_bar=True
    )
        
@error_log
@timer
def initial_labeling(data: pd.Series) -> pd.DataFrame: 
    def text_pipe(texts: pd.DataFrame, batch_size=16):
        results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            truncated_batch = []
            for text in batch:
                tokens = _tokenizer(
                    text,
                    max_length=512,
                    truncation=True
                )
                truncated_batch.append(
                    _tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
                )
                
            results.extend(_pipe(truncated_batch))
        return results
    
    prediction = text_pipe(data.to_list(), cfg.models.spam_detection.batch_size) 
    label = LabelEncoder().fit_transform([p['label'] for p in prediction])
    score = [p['score'] for p in prediction]
    
    return pd.DataFrame({
        cfg.data.target_column: data,
        cfg.data.target_column + '_label': label,
        cfg.data.target_column + '_score': score
    })
        