from typing import List, Any 
from loader.config_loader import get_config 

import numpy as np
import errno
import faiss
import pickle
import os
import gc 

class VectorStore:
    def __init__(self, dimension: int): 
        self.dimension = dimension
        self.index = None
     
    def write(self, embeddings: np.ndarray):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
           
    def load_exist_index(self, db_path: str):
        try:
            self.index = faiss.read_index(index_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Faiss index file not found: {index_path}')
      
    def save_to(self, destination: str):
        try:
            faiss.write_index(self.index, destination)
        except Exception as e:
            raise Exception(e)
     
    def similarity_search(self, embedding: np.ndarray, n: int = 5):
        return self.index.search(embedding, n)
     
    def get_index(self):
        return self.index
    
    def close(self):
        del self.index
        gc.collect()