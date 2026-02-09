from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from tqdm import tqdm
from scipy import stats

import os
import numpy as np
import pandas as pd


class VectorStore: 
    def __init__(self, directory: str, filename: str, embedding: HuggingFaceEmbeddings):
        self.directory = directory 
        self.filename = filename 
        self.embedding = embedding
        self.faiss = None
     
    
    @task(name='Load saved index from local file', cache_policy=NO_CACHE)
    def load_local_file(self):
        """Load local file
        
        Raises:
            TypeError: if parameter name is wrong, parameter in wrong type or missing
            FileNotFoundError: if self.directory is not created
            Exception: any other unexpected error
        """
        
        logger = get_run_logger()
        
        try:
            if self.filename + '.pkl' in os.listdir(self.directory):
                logger.info('Loading local index')
                self.faiss = FAISS.load_local( 
                    folder_path=self.directory, 
                    index_name=self.filename, 
                    embeddings=self.embedding,
                    allow_dangerous_deserialization=True
                )

        except TypeError as e:
            logger.error(f'Invalid parameter: {e}', exc_info=True)
            raise
        except FileNotFoundError as e:
            logger.error(f'File not found: {e}', exc_info=True)
            raise
            
    
    @task(name='Initial vectorstore with dataframme', cache_policy=NO_CACHE)
    def load_with_dataframe(self, data: pd.DataFrame, save: bool = True):
        """Setup a base faiss index

        Args:
            data (pd.DataFrame): base embeddings

        Raises:
            TypeError: if parameter is not dataframe or missing 
        """

        logger = get_run_logger()

        try: 
            documents = [
                Document(page_content=payload, metadata={'label': label, 'document_id': i})
                for i, (payload, label) in enumerate(zip(*data.T.to_numpy()))
            ]
            
            self.faiss = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding
            )
            
            if save:
                self.faiss.save_local(folder_path=self.directory, index_name=self.filename)
                 
        except TypeError as e:
            logger.error(f'Invalid data type or missing parameter: {e}', exc_info=True)
            raise


    @task(name='Return embeddings and labels', cache_policy=NO_CACHE)
    def return_pagecontent_embeddings_and_labels(self) -> tuple[pd.Series, np.ndarray, np.ndarray]:
        """Retrieve all page content, embeddings and labels from vectorstore

        Raises:
            KeyError: when trying to access a unknown dict key 

        Returns:
            tuple[pd.Series, np.ndarray, np.ndarray]: all embeddings, labels in vectorstore
        """
        
        logger = get_run_logger()
        
        try:
            embeddings = self.faiss.index.reconstruct_n(0, -1)  
            
            n = self.faiss.index.ntotal
            
            labels = np.empty(n, dtype=int) 
            payload = np.empty(n, dtype=object)
            for doc in self.faiss.docstore._dict.values():
                payload[doc.metadata['document_id']] = doc.page_content
                labels[doc.metadata["document_id"]] = doc.metadata['label']
    
            return pd.Series(payload), embeddings, labels
    
        except KeyError as e:
            logger.error(f'Invalid key: {e}', exc_info=True)
            raise


    @task(name="Label uncertain data", cache_policy=NO_CACHE)
    def label_uncertains(self, sentences: list) -> tuple[np.ndarray, np.ndarray]:
        """Label uncertain embeddings with vectorstore's similarity search

        Args:
            uncertain_embeddings (np.ndarray): uncertain embeddings for model to label

        Raises:
            TypeError: if parameter's name is wrong, type is wrnog or missing

        Returns:
            tuple[np.ndarray, np.ndarray]: list of binary labels
        """
         
        logger = get_run_logger()
         
        labels_status = []
        labels = []
        
        try:
            for sentence in tqdm(sentences):
                top_k_labels = []
                
                docs = self.faiss.similarity_search(query=sentence, k=50)
                for doc in docs:
                    top_k_labels.append(doc.metadata['label'])
                
                labels.append(stats.mode(np.asarray(top_k_labels)).mode)
                
                counts = np.bincount(top_k_labels) / len(top_k_labels)
                entropy = -1 * np.sum([p * np.log2(p) for p in counts])
                
                if entropy < 0.8:
                    labels_status.append(-1)
                else:
                    labels_status.append(1)
                    
            return np.asarray(labels_status), np.asarray(labels)
        
        except KeyError as e:
            logger.error(f'Invalid key: {e}', exc_info=True)
            raise
        except TypeError as e:
            logger.error('Invalid parameter: {e}', exc_info=True)
            raise