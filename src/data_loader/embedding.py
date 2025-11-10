from prefect import task 
from prefect.cache_policies import NO_CACHE
from typing import List, Any 
from sentence_transformers import SentenceTransformer

import os
import numpy as np
import pandas as pd


class Embedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
    @task(cache_policy=NO_CACHE)
    def embed_message(self, messages: pd.DataFrame, target_column: str, batch_size=4) -> np.ndarray:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return self.model.encode(messages[target_column], batch_size=batch_size, show_progress_bar=True)
    
    def get_embeddings(self):
        return self.model