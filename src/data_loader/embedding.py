from prefect import task 
from prefect.cache_policies import NO_CACHE
from typing import List, Any 
from sentence_transformers import SentenceTransformer

import os
import numpy as np
import pandas as pd


class EmbeddingPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
    @task(cache_policy=NO_CACHE)
    def embed_message(self, messages: List[Any]) -> np.ndarray:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        return self.model.encode(messages, batch_size=4)