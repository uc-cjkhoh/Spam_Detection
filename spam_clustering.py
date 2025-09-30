import os
import re
import ftfy
import emoji
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from yellowbrick.cluster import KElbowVisualizer

from loader.config_loader import cfg 


def get_optimal_n_cluster(data: pd.Series, cluster_limit):
    kmeans = KMeans(init='k-means++')
    visualizer = KElbowVisualizer(kmeans, k=(1, cluster_limit))    
    visualizer.fit(data)
    
    return visualizer.elbow_value_


def text_normalize(data: pd.Series):
    """
    Normalize message structure

    Args:
        data (pd.DataFrame): data 

    Returns:
        pd.DataFrame: add two columns (decoded_message, decoded_message_length)
    """
      
    try:
        # data[cfg.data.target_column] = data[cfg.data.target_column].apply(ftfy.fix_text)
        data = data.apply(str.strip) 
        data = data.apply(lambda x: re.sub('\s+', ' ', x))
        data = data.apply(lambda x: x.replace('\n', ' '))
        data = data.apply(lambda x: emoji.replace_emoji(x, '<EMO>'))
         
        if cfg.data.drop_null:
            data = data.dropna()
        if cfg.data.drop_duplicates:
            data = data.drop_duplicates()
             
        return data
    except KeyError:
        print('Invalid column, check if column_name and payload_column is the same in ./configs/config.yaml')      
        

def main(): 
    folder = 'data/label' 
    data = pd.read_excel(os.path.join(folder, '0.xlsx')).sample(frac=0.3)
    data = data[data['message_label'] == 1]
    
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
       
    normalized_message = text_normalize(data['message'].copy().astype(str))
    embedded_text = model.encode(normalized_message.to_numpy(), batch_size=2, show_progress_bar=True)
    
    # pca = PCA(n_components=3)
    # supppress_data = pca.fit_transform(embedded_text)
    
    optimal_n_cluster = get_optimal_n_cluster(embedded_text, cluster_limit=20)
    kmeans = KMeans(n_clusters=optimal_n_cluster)
    initial_label = kmeans.fit_predict(embedded_text)
    
    svm = SVC()
    svm.fit(embedded_text, initial_label)
    category = svm.predict(embedded_text)
    
    data['category'] = category
    
    to_folder = 'testing'
    data.to_excel(os.path.join(to_folder, '0.xlsx'))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)