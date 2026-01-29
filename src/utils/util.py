import os  
import sys
import argparse
import numpy as np
import pandas as pd  

from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import RobustScaler 
from langchain.schema import Document 
from langchain_huggingface import HuggingFaceEmbeddings 

from src.ml.model_training import SGD 
from src.data_loader.database import Database
from src.config_folder.config_loader import get_config
from src.vector_database.vectorstore import VectorStore 
from src.data_loader.preprocessing import get_normalized_messages

from prefect import task
from prefect.cache_policies import NO_CACHE 


@task(name='Setup Environment', cache_policy=NO_CACHE)
def setup_core_components(args) -> tuple[dict, Database, HuggingFaceEmbeddings, VectorStore, SGD, SGD]: 
    """Setup require directories, files, and components (models, vectorstore)

    Args:
        args (argparse.Namespace): terminal arguments

    Returns:
        tuple[dict, Database, HuggingFaceEmbeddings, VectorStore, SGD, SGD]: 
        local configuration, connected mysql, faiss, SGDClassifier, SGDClassifier
    """
    
    config = get_config() 
    
    os.makedirs('./data/vector', exist_ok=True) 
      
    # 1. setup connection to mysql
    database = Database(
        host="10.168.51.196",
        port=3306,
        user='unified',
        password='unified'
    )
    
    # 2. setup embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={'trust_remote_code': True},
        encode_kwargs={
            'batch_size': 8,
            'normalize_embeddings': True
        },
        show_progress=True
    )
    
    # 3. setup vectorstore
    vectorstore = VectorStore(
        directory='./data/vector', 
        filename=f'sms_embeddings_{str.lower(args.experiment)}',
        embedding=embedding_model
    )
    
    # 4. setup models
    teacher = SGD(experiment_name=args.experiment, model_name='Teacher') 
    student = SGD(experiment_name=args.experiment, model_name='Student')
       
    # 5. get initial data
    initial_data = database.get_records(config.initial_data)
    
    # 6. terminate if the labels are not done yet
    if initial_data[args.target_column].isna().sum():  
        sys.exit(f"Please finish labelling the data in table `sms_spam_cd.initial_data`")

    # 7. create or load vector database
    if vectorstore.filename + '.pkl' in os.listdir(vectorstore.directory):
        # retrieve embeddings from vectorstore
        vectorstore.load_index(folder_path=vectorstore.directory, index_name=vectorstore.filename)
    else:
        # create new vectorstore and store initial data
        labels = initial_data.pop(args.target_column)
        features = get_normalized_messages(initial_data, config.data.target_column)
        payloads = features.pop(config.data.target_column)
        
        features.to_csv(f'./data/initial_data_features_{str.lower(args.experiment)}.csv', index=False)
        
        documents = [
            Document(page_content=payload, metadata={'label': label, "faiss_id": i})
            for i, (payload, label) in enumerate(zip(payloads, labels))
        ] 
        
        vectorstore.write_index(documents)
       
    return config, database, embedding_model, vectorstore, teacher, student
 
 
@task(name='Stratified Sampling', cache_policy=NO_CACHE)
def stratified_sampling(config, db) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Stratified Sampling

    Args:
        config (PyYAML): local configuration
        db (Database): database object

    Returns:
        tuple[pd.Series, pd.Series, pd.DataFrame]: id, datetime, payload from MySQL 
    """
    
    data = db.get_records(config.data.query)
    return data['id'], data['current_datetime'], data[['payload']]
 

@task(name='Feature Engineering and Sentence Embeddings', cache_policy=NO_CACHE)
def preprocess_data(embedding_model: HuggingFaceEmbeddings, sms_message: pd.DataFrame, target_column: str) -> np.ndarray:  
    """Extract features from raw text and perform transformation such as text normalization and cleaning

    Args:
        embedding_model (HuggingFaceEmbeddings): Transformer to convert text to embeddings
        sms_message (pd.DataFrame): DataFrame that contain the payloads or messages
        target_column (str): the column name for the payloads or messages

    Returns:
        np.ndarray: embeddings
    """
    
    features = get_normalized_messages(sms_message, target_column=target_column) 
    
    messages = features.pop(target_column)
    embeddings = np.asarray(embedding_model.embed_documents(messages))   
    
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    
    embeddings = np.hstack((features, embeddings))
    return embeddings 


@task(name='Oversampling', cache_policy=NO_CACHE)
def oversampling(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perform oversampling

    Args:
        x (np.ndarray): independent features
        y (np.ndarray): dependent feature

    Returns:
        tuple[np.ndarray, np.ndarray]: oversampled x and y
    """
    
    smote = SMOTE(random_state=42)
    resampled_x, resampled_y = smote.fit_resample(x, y)    
    return resampled_x, resampled_y


@task(name='Load train data', cache_policy=NO_CACHE)
def load_train_data(args: argparse.Namespace, config: dict, database: Database, 
                    embedding_model: HuggingFaceEmbeddings, vectorstore: VectorStore) -> tuple[np.ndarray, np.ndarray]: 
    """Load training data

    Args:
        args (argparse.Namespace): terminal arguments
        config (dict): local configuration
        database (Database): connected database (MySQL)
        embedding_model (HuggingFaceEmbeddings): embed_model
        vectorstore (VectorStore): vectorstore (faiss)

    Returns:
        tuple[np.ndarray, np.ndarray]: embeddings, labels
    """
    
    @task(name='Load initial data', cache_policy=NO_CACHE)
    def load_initial_data():
        """Load initial data

        Returns:
            np.ndarray: robust-scaled embeddings
            np.ndarray: corresponding labels
        """
        
        scaler = RobustScaler()
        
        initial_embeddings = vectorstore.faiss.index.reconstruct_n(0, -1)  
        initial_features = pd.read_csv(f'./data/initial_data_features_{str.lower(args.experiment)}.csv')
        initial_features = scaler.fit_transform(initial_features)
        
        scaled_initial_embeddings = np.hstack((initial_features, initial_embeddings))

        n = vectorstore.faiss.index.ntotal
        labels = np.empty(n, dtype=int) 
        for doc in vectorstore.faiss.docstore._dict.values():
            labels[doc.metadata["faiss_id"]] = doc.metadata['label']
 
        return scaled_initial_embeddings, labels
   
   
    @task(name='Load MySQL data', cache_policy=NO_CACHE)
    def load_mysql_data():
        """Load human labeled data in MySQL

        Returns:
            np.ndarray: robust-scaled embeddings
            np.ndarray: corresponding labels
        """ 
        
        mysql_data = database.get_records(config.labeled_data)
        
        if len(mysql_data) == 0:
            return np.array([]), np.array([])
        
        mysql_data_labels = mysql_data.pop(args.target_column)
        mysql_data_embeddings = preprocess_data(
            embedding_model=embedding_model,
            sms_message=mysql_data, 
            target_column=config.data.target_column
        ) 
        
        return mysql_data_embeddings, mysql_data_labels
   
   
    initial_embeddings, initial_labels = load_initial_data()
    mysql_embeddings, mysql_labels = load_mysql_data()

    if len(mysql_embeddings) == 0: 
        x_train, y_train = oversampling(initial_embeddings, initial_labels)
        return x_train, y_train
    else:
        # combine initial data with sql data
        complete_train_embeddings = np.vstack((initial_embeddings, mysql_embeddings)) 
        complete_train_labels = np.hstack((initial_labels, mysql_labels)).reshape(-1, 1)
        x_train, y_train = oversampling(complete_train_embeddings, complete_train_labels)
        return x_train, y_train


@task(name='Load test data', cache_policy=NO_CACHE)
def load_test_data(args: argparse.Namespace, config: dict, database: Database, embedding_model: HuggingFaceEmbeddings):
    """Load test data (for model evaluation)

    Args:
        args (argparse.Namespace): terminal arguments 
        config (dict): local configuration
        database (Database): connected database
        embedding_model (HuggingFaceEmbeddings): embedding transformer

    Raises:
        ValueError: if the table has no data

    Returns:
        tuple[np.ndarray, np.ndarray]: testing embeddings and labels 
    """
    
    test_data = database.get_records(config.test_data) 
    
    if test_data.shape[0] > 0:
        y_test = test_data.pop(args.target_column).to_numpy().squeeze() 
        x_test = preprocess_data(embedding_model, test_data, target_column=config.data.target_column) 
        return x_test, y_test
    else:
        raise ValueError('Missing test data in sql table')
 
