from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings

from prefect import task
from prefect.cache_policies import NO_CACHE 


class VectorStore:
    @task(name="Create Vector Database Class", cache_policy=NO_CACHE)
    def __init__(self, directory: str, filename: str, embedding: HuggingFaceEmbeddings):
        self.directory = directory 
        self.filename = filename 
        self.embedding = embedding
        self.faiss = None
     
     
    @task(name="Load Faiss index", cache_policy=NO_CACHE)
    def load_index(self, folder_path: str, index_name: str):
        """Load existing faiss index

        Args:
            folder_path (str): folder name
            index_name (str): filename of the faiss index
        """
        
        self.faiss = FAISS.load_local(
            folder_path=folder_path, 
            index_name=index_name, 
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )
        
        
    @task(name="Write Into Faiss", cache_policy=NO_CACHE)
    def write_index(self, documents: list):
        """Write documents to Faiss

        Args:
            documents (list): list of Document (langchain)
        """
        
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
        """Save Faiss index

        Raises:
            ValueError: if self.faiss is None object
        """
        
        if self.faiss is not None:
            self.faiss.save_local(folder_path=self.directory, index_name=self.filename)
        else:
            raise ValueError("Cannot save empty index")
