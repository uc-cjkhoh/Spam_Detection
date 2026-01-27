from langchain_community.vectorstores import FAISS 
 
from prefect import task
from prefect.cache_policies import NO_CACHE 


class VectorStore:
    @task(name="Create Vector Database Class", cache_policy=NO_CACHE)
    def __init__(self, directory, filename, embedding):
        self.directory = directory 
        self.filename = filename 
        self.embedding = embedding
        self.faiss = None
     
     
    @task(name="Load Faiss index", cache_policy=NO_CACHE)
    def load_index(self, folder_path, index_name):
        self.faiss = FAISS.load_local(
            folder_path=folder_path, 
            index_name=index_name, 
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )
        
        
    @task(name="Write Into Faiss", cache_policy=NO_CACHE)
    def write_index(self, documents):
        if self.faiss is None:
            self.faiss = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding
            )
        else:
            pass
        
        self.save()
    
    
    @task(name="Save Faiss Index", cache_policy=NO_CACHE)
    def save(self):
        if self.faiss is not None:
            self.faiss.save_local(folder_path=self.directory, index_name=self.filename)
        else:
            raise ValueError("Cannot save empty index")
