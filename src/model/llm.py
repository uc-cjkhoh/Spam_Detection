import pandas as pd 
import numpy as np 

from tqdm import tqdm  
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer 
from prefect import task
from prefect.cache_policies import NO_CACHE
 

@task(cache_policy=NO_CACHE)
def label_by_pretrained_llm(data: pd.Series, cfg: dict) -> list, list: 
    def text_pipe(texts: pd.DataFrame, batch_size=16):
        detector_tokenizer = AutoTokenizer.from_pretrained(cfg.models.spam_detection.model_name)
        pipe = pipeline('text-classification', model=cfg.models.spam_detection.model_name)
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            truncated_batch = []
            for text in batch:
                tokens = detector_tokenizer(
                    text,
                    max_length=512,
                    truncation=True
                )
                truncated_batch.append(detector_tokenizer.decode(tokens["input_ids"], skip_special_tokens=True))
                
            results.extend(pipe(truncated_batch))
        return results
    
    prediction = text_pipe(data.to_list(), cfg.models.spam_detection.batch_size) 
    label = LabelEncoder().fit_transform([p['label'] for p in prediction])
    score = [p['score'] for p in prediction]
    
    return label, score
        