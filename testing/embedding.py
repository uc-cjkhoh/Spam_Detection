from typing import List, Any
from sentence_transformers import SentenceTransformer

import os
import numpy as np


class EmbeddingPipeline:
    def __init__(self, model: str):
        self.model = model
        
    def embed_message(self, messages: List[Any]) -> np.ndarray:
        embedding_model = SentenceTransformer(self.model)
        embeddings = embedding_model.encode(messages)
        return embeddings