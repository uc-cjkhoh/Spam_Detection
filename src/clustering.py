import os
import faiss 
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

from src.decorators import error_log, timer 
from loader.config_loader import cfg
from loader.data_loader import Database


@error_log
@timer
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
    
    hdbscan = HDBSCAN(min_cluster_size=5)
    cluster_label = hdbscan.fit_predict(reduced_vectors)
    return cluster_label


class HNSW:
    def __init__(self, cur):
        self.cur = cur
        self.hnsw_index = None
        self.cluster_centroids = {}
        self.cluster_labels = []
    
    # execute when initializing the first subdate/model
    def initial(self, vectors):
        hdbscan = HDBSCAN(min_cluster_size=5)
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
        for vector in tqdm(vectors):
            _, indices = self.hnsw_index.search(vector, k)
            
            # select vector by indices
            where_statement = str(tuple(indices[0].tolist()))
            complete_query = cfg.hnsw.query_by_indices.format(where_statement)
            
            self.cur.execute(complete_query)
            nearest_label = np.array(self.cur.fetchall())
            
            # vote to cluster
            most_vote = stats.mode(nearest_label)
            
            # add label to `cluster_label`
            cluster_label.append(most_vote)
            
            # update hnsw
            self.hnsw_index.add(vector)
            
        self.save_hnsw(destination=os.path.join(cfg.hnsw.folder, cfg.hnsw.filename))
        return np.array(cluster_label)
      
    def save_hnsw(self, destination):
        os.makedirs(destination, exist_ok=True)
        faiss.write_index(self.hnsw_index, destination)
    
     