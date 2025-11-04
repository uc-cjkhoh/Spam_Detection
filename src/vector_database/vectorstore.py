import numpy as np 
import faiss 
import gc 
import os

from prefect import task
from prefect.cache_policies import NO_CACHE

  
class VectorStore:
    def __init__(self, config): 
        self.config = config 
        self.filepath = os.path.join(self.config.directory, self.config.filename)
        self.index = self.load_any()
     
    def load_any(self):
        if os.path.exists(self.filepath):
            self.index = faiss.read_index(self.filepath)
        else:
            self.index = None

    @task(cache_policy=NO_CACHE)    
    def write(self, embeddings: np.ndarray):
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1]) 
        self.index.add(embeddings)
        
    def similarity_search(self, embedding: np.ndarray, n: int = 5):
        return self.index.search(embedding, n)
     
    def get_vectors(self):
        return self.index
    
    def get_vectorstore_filepath(self):
        return self.filepath
    
    def save(self):
        try:
            faiss.write_index(self.index, self.filepath)
        except Exception as e:
            raise Exception(e)
     
    def close(self):
        del self.index
        gc.collect()