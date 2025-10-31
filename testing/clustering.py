import os
import faiss 
import ast
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from src.decorators import error_log, timer 
from loader.config_loader import cfg
from loader.data_loader import get_connector

 
def direct_clustering(vectors):
    """
    Directly cluster vectors parameter

    Args:
        vectors (np.ndarray): any multi-dimensional array 

    Returns:
        np.ndarray: signed integer labelling for each vector
    """
    
    # reduce dimension
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(vectors)
    
    hdbscan = HDBSCAN(min_cluster_size=10)
    cluster_label = hdbscan.fit_predict(reduced_vectors)
    return cluster_label

 
def recluster_from_database(): 
    con = get_connector()
    cur = con.cursor()
    
    cur.execute(cfg.reclustering.select_query) 
    data = cur.fetchall()
    
    embeddings = [] 
    idxs = []
    for row in tqdm(data):
        idx = row[0]
        idxs.append(idx)
        
        embedding = row[1].decode('utf-8')
        embeddings.append(ast.literal_eval(embedding))
        
    embeddings = np.array(embeddings) 
    labels = direct_clustering(embeddings)
    
    values = [[int(_label), _idx] for _label, _idx in zip(labels, idxs)]
    # print(values[:100])
    cur.executemany(cfg.reclustering.update_query, values)
    con.commit()
    
    exist_hnsw = faiss.read_index(os.path.join(cfg.hnsw.folder, cfg.hnsw.filename))
    exist_hnsw.reset()
    exist_hnsw.write_index(embeddings)
    

class HNSW:
    def __init__(self):
        self.hnsw_index = None
        self.cluster_centroids = {}
        self.cluster_labels = []
    
    # execute when initializing the first subdate/model
    def initial(self, vectors):
        hdbscan = HDBSCAN(min_cluster_size=10)
        self.cluster_labels = hdbscan.fit_predict(vectors)
        self._calculate_centroids(vectors) 
        self._build_hnsw(vectors)
    
    # excute when iterate through each subdata
    def load_index(self, target_filepath):
        try:
            self.hnsw_index = faiss.read_index(target_filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f'{target_filepath} does not exists')
        
    def _calculate_centroids(self, vectors):
        for label in np.unique(self.cluster_labels):
            if label == -1:
                continue
            
            cluster_points = vectors[self.cluster_labels == label]
            self.cluster_centroids[label] = cluster_points.mean(axis=0)
    
    def _build_hnsw(self, vectors):
        self.hnsw_index = faiss.IndexHNSWFlat(vectors.shape[-1], 32) 
        self.hnsw_index.efSearch = 2 * 32
        self.hnsw_index.add(vectors)
          
    def cluster_and_save(self, vectors, k=5):  
        cluster_label = []
        # Ensure vectors is 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif len(vectors.shape) > 2:
            vectors = vectors.reshape(vectors.shape[0], -1)
            
        for vector in tqdm(vectors):
            # Reshape single vector to 2D array for FAISS
            vector = vector.reshape(1, -1)
            _, indices = self.hnsw_index.search(vector, k)
            
            # select vector by indices
            indices_list = indices[0].tolist()
            where_statement = f"({indices_list[0]})" if len(indices_list) == 1 else str(tuple(indices_list))
            complete_query = cfg.hnsw.query_by_indices.format(where_statement)
            
            con = get_connector()
            cur = con.cursor()
            
            cur.execute(complete_query)
            nearest_label = np.array([row[0] for row in cur.fetchall()])
            
            # Handle empty results
            if len(nearest_label) == 0:
                cluster_label.append(-1)
                self.hnsw_index.add(vector)
                continue
                
            # vote to cluster using updated scipy.stats.mode
            try:
                mode_result = stats.mode(nearest_label)
                most_common = mode_result.mode[0] if hasattr(mode_result, 'mode') else mode_result[0]
            except:
                most_common = -1  # fallback if mode calculation fails
                
            # add label to `cluster_label`
            cluster_label.append(most_common)
            
            # update hnsw with reshaped vector
            self.hnsw_index.add(vector)
            
        self.save_hnsw(destination=os.path.join(cfg.hnsw.folder, cfg.hnsw.filename))
        return np.array(cluster_label)
      
    def save_hnsw(self, destination):
        faiss.write_index(self.hnsw_index, destination)

