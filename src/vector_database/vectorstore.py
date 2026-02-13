from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from tqdm import tqdm
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from data_validation.vectorstore_validation.validate_vectorstore import VectorstoreConfig

import os
import numpy as np
import pandas as pd


class VectorStore: 
    def __init__(self, vectorstore_config: VectorstoreConfig, embedding: HuggingFaceEmbeddings):
        """Initiate vectorstore

        Args:
            vectorstore_config (VectorstoreConfig): refer to VectorstoreConfig in validate_vectorstore.py
        """
        
        self.directory = vectorstore_config.directory 
        self.filename = vectorstore_config.filename 
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
            data (pd.DataFrame): data to saved
            save (bool): whether to save index

        Raises:
            TypeError: if parameter is not dataframe or missing 
        """
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Data need to be in pandas Dataframe type')
        elif not isinstance(save, bool):
            raise TypeError('Save attribute need to be boolean type')

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
    def label_uncertains(self, embeddings: np.ndarray, similarity_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """Label uncertain embeddings with vectorstore's similarity search

        Args:
            embeddings (np.ndarray): uncertain embeddings for model to label
            similarity_threshold (float): threshold to label them with confidence

        Raises:
            TypeError: if parameter's name is wrong, type is wrnog or missing

        Returns:
            tuple[np.ndarray, np.ndarray]: list of binary labels
        """
        
        if not isinstance(embeddings, np.ndarray):
            raise TypeError('Sentences need to be in list type')
         
        logger = get_run_logger()
         
        labels_status = [-1] * len(embeddings)
        labels = [-1] * len(embeddings)
        
        try:
            for i, embedding in tqdm(enumerate(embeddings)):
                current_embedding = np.asarray(embedding).squeeze()
                
                nearest_doc = self.faiss.similarity_search_by_vector(current_embedding, k=1)
                
                id = nearest_doc[0].id
                nearest_label = nearest_doc[0].metadata['label']
                nearest_doc_index = 0
                
                for index, item_id in self.faiss.index_to_docstore_id.items():
                    if item_id == id:
                        nearest_doc_index = index
                
                nearest_embeddings = self.faiss.index.reconstruct(nearest_doc_index)
                
                similarity_score = cosine_similarity(nearest_embeddings.reshape(1, -1), current_embedding.reshape(1, -1))
                
                if np.asarray(similarity_score).squeeze() > similarity_threshold: 
                    labels[i] = nearest_label
                    labels_status[i] = 1
                else:
                    labels_status[i] = -1
                    
            return labels_status, labels
        
        except KeyError as e:
            logger.error(f'Invalid key: {e}', exc_info=True)
            raise
        except TypeError as e:
            logger.error('Invalid parameter: {e}', exc_info=True)
            raise