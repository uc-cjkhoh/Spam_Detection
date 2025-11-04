import numpy as np 
import faiss 
import gc 
import os

from testing.config_loader.config_loader import get_config

class VectorStore:
    def __init__(self): 
        self.vector_config = get_config().vectorstore 
        self.filepath = os.path.join(self.vector_confg.directory, self.vector_config.filename)
        self.vector = None
     
    def write(self, embeddings: np.ndarray):
        if self.vector is None:
            self.vector = faiss.IndexFlatL2(embeddings.shape[1])
            
        self.vector.add(embeddings) 
        
    def save(self):
        faiss.write_index(self.vector, self.filepath)
     
    def similarity_search(self, embedding: np.ndarray, n: int = 5):
        return self.vector.search(embedding, n)
     
    def get_vectors(self):
        return self.vector
    
    def get_vectorstore_filepath(self):
        return self.filepath
    
    def close(self):
        del self.vector
        gc.collect()