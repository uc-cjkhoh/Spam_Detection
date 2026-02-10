from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from prefect import task, get_run_logger
from prefect.cache_policies import NO_CACHE 

from tqdm import tqdm
from scipy import stats
from data_validation.vectorstore_validation.validate_vectorstore import VectorstoreConfig, \
    LoadDataFrameInput, LabelUncertainConfig

import os
import numpy as np
import pandas as pd


class VectorStore: 
    def __init__(self, vectorstore_config: VectorstoreConfig):
        """Initiate vectorstore

        Args:
            vectorstore_config (VectorstoreConfig): refer to VectorstoreConfig in validate_vectorstore.py
        """
        
        self.directory = vectorstore_config.directory 
        self.filename = vectorstore_config.filename 
        self.embedding = vectorstore_config.embedding
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
    def load_with_dataframe(self, load_df_input: LoadDataFrameInput):
        """Setup a base faiss index

        Args:
            load_df_input (pd.LoadDataFrameInput): refer to LoadDataFrameInput in validate_vectorstore.py

        Raises:
            TypeError: if parameter is not dataframe or missing 
        """

        logger = get_run_logger()

        try: 
            documents = [
                Document(page_content=payload, metadata={'label': label, 'document_id': i})
                for i, (payload, label) in enumerate(zip(*load_df_input.data.T.to_numpy()))
            ]
            
            self.faiss = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding
            )
            
            if load_df_input.save:
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
    def label_uncertains(self, label_uncertain_config: LabelUncertainConfig) -> tuple[np.ndarray, np.ndarray]:
        """Label uncertain embeddings with vectorstore's similarity search

        Args:
            label_uncertain_config (LabelUncertainConfig): uncertain embeddings for model to label

        Raises:
            TypeError: if parameter's name is wrong, type is wrnog or missing

        Returns:
            tuple[np.ndarray, np.ndarray]: list of binary labels
        """
         
        logger = get_run_logger()
         
        labels_status = []
        labels = []
        
        try:
            for sentence in tqdm(label_uncertain_config.sentences):
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