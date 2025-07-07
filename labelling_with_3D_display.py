import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import plotly.express as px
import statistics as st
import umap 
import utils
from sentence_transformers import SentenceTransformer  
from sklearn.cluster import KMeans 
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
 
# main 
if __name__  == '__main__':
    # load data from spam.csv file
    ori_data = pd.read_csv(r'C:\Users\cj_khoh\Documents\UnifiedComms\Data\spam.csv', encoding='ISO-8859-1')
    data = ori_data

    # initialize sentence embedding model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # clean text 
    data['v2'] = utils.clean_text(data['v2'])
    
    # separate text and label, then perform text embedding
    sent, label = data['v2'], data['v1']
    encoded_sent = model.encode(sent, show_progress_bar=True)  

    # perform over-sampling
    X, Y = RandomOverSampler().fit_resample(encoded_sent, label)

    # reduce dimension    
    umap_sent = umap.UMAP(n_neighbors=10, min_dist=0, n_components=3, metric='cosine').fit_transform(X) 
    
    # group all data point to 12 clusters
    cluster_model = KMeans(n_clusters=12, init='k-means++', n_init=30) 
    y = cluster_model.fit_predict(umap_sent)

    # distinguish 'Spam' and 'Not spam' data point
    pred = np.array(y)
    pred = [0 if x == st.mode(pred) else 1 for x in pred]
    pred = pd.DataFrame(pred, columns=['type'])

    sent = pd.DataFrame(X, columns=['v' + str(i) for i in range(len(X[0]))])
    
    output = pd.concat([pred, sent], axis=1, ignore_index=True)
    output.to_csv('./labeled_spam.csv', index=False)

    print(f'Accuracy for clustering 1: %.3f' % accuracy_score(pd.factorize(Y)[0], pred))
 
    #Visualize
    fig = px.scatter_3d(
        umap_sent, x=0, y=1, z=2,
        color=Y, labels={'color': "MSG TYPE"}
    )
    # fig.update_traces(marker_size=8)
    fig.show()  

    fig = px.scatter_3d(
        umap_sent, x=0, y=1, z=2,
        color=y, labels={'color': "MSG TYPE"}
    )
    # fig.update_traces(marker_size=8)
    fig.show() 
