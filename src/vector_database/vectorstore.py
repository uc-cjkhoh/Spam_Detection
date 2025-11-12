import os

from langchain_community.vectorstores import FAISS
from prefect import task
from prefect.cache_policies import NO_CACHE


class VectorStore:
    def __init__(self, config): 
        self.config = config 
        self.filepath = os.path.join(self.config.directory, self.config.filename)
        self.index = self._check_any_existing_vectorstore()
    
    @task(cache_policy=NO_CACHE)
    def _check_any_existing_vectorstore(self):
        if os.path.exists(self.filepath):
            self.index = FAISS.load_local(self.filepath)
        else:
            self.index = None

    @task(cache_policy=NO_CACHE)
    def write_to_vectorstore(self, text_embedding_pair, embedding, metadatas: list[dict]):
        if self.index is None:
            self.index = FAISS.from_embeddings(
                text_embeddings=text_embedding_pair,
                embedding=embedding,
                metadatas=metadatas
            )
        else:
            self.index.add_embeddings(
                text_embeddings=text_embedding_pair,
                metadatas=metadatas
            )
        
        self.save()
        
    def save(self):
        if self.index is not None:
            self.index.save_local(folder_path=self.config.directory, index_name=self.config.filename)
        else:
            raise ValueError("Cannot save empty index")

    def get_index(self):
        return self.index