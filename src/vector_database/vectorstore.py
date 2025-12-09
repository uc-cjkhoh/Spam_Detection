import os

from langchain_community.vectorstores import FAISS
from prefect import task
from prefect.cache_policies import NO_CACHE


class VectorStore:
    def __init__(self, directory, filename, embedding):
        self.directory = directory 
        self.filename = filename 
        self.embedding = embedding
        self.index = self._check_any_existing_vectorstore()
     
    def _check_any_existing_vectorstore(self):
        filepath = os.path.join(self.directory, self.filename)
        if os.path.exists(filepath):
            self.index = FAISS.load_local(
                folder_path=self.directory, 
                index_name=self.filename, 
                embeddings=self.embedding,
                allow_dangerous_deserialization=True
            )
        else:
            self.index = None

    @task(cache_policy=NO_CACHE)
    def write_to_vectorstore(self, text_embedding_pair, embedding, metadatas: list[dict] = None):
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
            self.index.save_local(folder_path=self.directory, index_name=self.filename)
        else:
            raise ValueError("Cannot save empty index")

    def get_index(self):
        return self.index