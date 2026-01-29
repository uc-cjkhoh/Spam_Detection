from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings

from prefect import task
from prefect.cache_policies import NO_CACHE 

from tqdm import tqdm 
from collections import Counter
import numpy as np


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


    @task(name="Label uncertain data", cache_policy=NO_CACHE)
    def label_uncertains(self, sentences: list) -> np.ndarray:
        """Label uncertain embeddings with vectorstore's similarity search

        Args:
            uncertain_embeddings (np.ndarray): uncertain embeddings for model to label

        Returns:
            np.ndarray: list of binary label (0, 1)
        """
        
        if self.faiss is None:
            raise ValueError("No existing faiss index")
        
        # # cut embeddings from 1049 dimension to 1024 dimension (remove from front)
        # embeddings = embeddings[:, embeddings.shape[-1] - 1024:]
        
        
        # docs = [
        #     self.faiss.similarity_search_by_vector(embedding, k=1)[0]
        #     for embedding in embeddings
        # ]
            
        # labels = [
        #     doc.metadata['label'] for doc in docs
        # ]
            
        # return np.asarray(labels)
        
        labels_status = []
        labels = []
        
        for sentence in tqdm(sentences):
            top_k_labels = []
            
            docs = self.faiss.similarity_search(query=sentence, k=50)
            for doc in docs:
                top_k_labels.append(doc.metadata['label'])
                
            labels.append(np.argmax(Counter(top_k_labels).values()))
            
            counts = np.bincount(top_k_labels) / len(top_k_labels)
            entropy = -1 * np.sum([p * np.log2(p) for p in counts])
            
            if entropy < 0.8:
                labels_status.append(-1)
                labels.append()
            else:
                labels_status.append(1)
                labels.append()