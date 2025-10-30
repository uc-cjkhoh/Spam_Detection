from typing import List, Any
from sentence_transformers import SentenceTransformer

import os
import numpy as np
import pandas as pd

class EmbeddingPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
    def embed_message(self, messages: pd.Series) -> np.ndarray:
        embeddings = self.model.encode(messages, batch_size=4, show_progress_bar=True, convert_to_numpy=True)
        return embeddings