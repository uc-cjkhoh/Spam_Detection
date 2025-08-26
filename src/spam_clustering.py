import os

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans

from .util import has_availabel_model
from .decorators import error_log, timer
from loader.config_loader import cfg

class SpamClustering:
    def __init__(self, spam_message: pd.DataFrame):
        self.model = MiniBatchKMeans()
        self.spam_message = spam_message
    
    def find_optimal_cluster_no(self):
        pass
        
    def run_cluatering(self):
        if has_availabel_model(type(self.model).__name__):
            pass
        else:
            pass
    
    def get_clusters(self):
        pass
    
    