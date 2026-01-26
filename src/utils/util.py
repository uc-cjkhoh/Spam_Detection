import os  
import sys
import numpy as np
import pandas as pd  

from sklearn.preprocessing import RobustScaler 
from imblearn.over_sampling import SMOTE 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.schema import Document 

from src.data_loader.connection import Database
from src.vector_database.vectorstore import VectorStore 
from src.ml.model_training import SGD 
from src.config_folder.config_loader import get_config
from src.data_loader.preprocessing import get_normalized_messages

from prefect import task
from prefect.cache_policies import NO_CACHE 


@task(name='Setup Environment', cache_policy=NO_CACHE)
def setup_core_components(args): 
    config = get_config() 
    
    os.makedirs('./data/vector', exist_ok=True)
    os.makedirs('./logs/evaluation', exist_ok=True)
     
    if not os.path.isfile('logs/evaluation/evaluation.xlsx'):
        pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Loss']).to_excel(
            'logs/evaluation/evaluation.xlsx', index=False
        ) 
    
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
        filename='sms_embeddings',
        embedding=embedding_model
    )
    
    # 4. setup models
    teacher = SGD(experiment_name=args.experiment, model_name='Teacher') 
    student = SGD(experiment_name=args.experiment, model_name='Student')
       
    # 5. build faiss index if not exists
    initial_data = database.get_records(
        f'select {config.data.target_column}, {args.target_column} from sms_spam_cd.initial_data where day(datetime) != 22', 
        columns=[config.data.target_column, args.target_column]
    ).squeeze()
    
    # terminate if the labels are not done yet
    if initial_data[args.target_column].isna().sum():  
        sys.exit(f"Please finish labelling the data in table `sms_spam_cd.initial_data`")

    # check if vectorstore created
    if len(os.listdir(vectorstore.directory)) > 0:
        # retrieve embeddings from vectorstore
        vectorstore.load_index(folder_path=vectorstore.directory, index_name=vectorstore.filename)
    else:
        # create new vectorstore and store initial data
        labels = initial_data.pop(args.target_column)
        features = get_normalized_messages(initial_data, config.data.target_column)
        payloads = features.pop(config.data.target_column)
        
        features.to_csv('./data/initial_data_features.csv', index=False)
        
        documents = [
            Document(page_content=payload, metadata={'label': label, "faiss_id": i})
            for i, (payload, label) in enumerate(zip(payloads, labels))
        ] 
        vectorstore.write_index(documents)
      
    x_train, y_train = load_train_data(args, config, database, embedding_model, vectorstore)
    x_test, y_test = load_test_data(args, config, database, embedding_model)
      
    return config, database, embedding_model, vectorstore, teacher, student, (x_train, y_train, x_test, y_test)
 
 
@task(name='Stratified Sampling', cache_policy=NO_CACHE)
def stratified_sampling(config, db):
    data = db.get_records(config.data.query, columns=config.data.column_name)
    return data['id'], data['datetime'], data[['payload']]
 

@task(name='Data Preprocessing', cache_policy=NO_CACHE)
def preprocess_data(embedding_model, sms_message: pd.DataFrame, target_column: str) -> np.ndarray:  
    features = get_normalized_messages(sms_message, target_column=target_column) 
    
    messages = features.pop(target_column)
    embeddings = np.asarray(embedding_model.embed_documents(messages))   
    
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    
    embeddings = np.hstack((features.to_numpy(), embeddings))
    return embeddings 


@task(name='Oversampling', cache_policy=NO_CACHE)
def oversampling(x, y):
    smote = SMOTE(random_state=42)
    resampled_x, resampled_y = smote.fit_resample(x, y)    
    return resampled_x, resampled_y


@task(name='Load train data', cache_policy=NO_CACHE)
def load_train_data(args, config, database, embedding_model, vectorstore): 
    @task(name='Load initial data', cache_policy=NO_CACHE)
    def load_initial_data():
        scaler = RobustScaler()
        
        initial_embeddings = vectorstore.faiss.index.reconstruct_n(0, -1)  
        initial_features = pd.read_csv('./data/initial_data_features.csv')
        initial_features = scaler.fit_transform(initial_features)
        
        scaled_embeddings = np.hstack((initial_features.to_numpy(), initial_embeddings))

        n = vectorstore.faiss.index.ntotal
        labels = np.empty(n, dtype=int)

        for doc in vectorstore.faiss.docstore._dict.values():
            labels[doc.metadata["faiss_id"]] = doc.metadata['label']
 
        return scaled_embeddings, labels
   
    @task(name='Load MySQL data', cache_policy=NO_CACHE)
    def load_mysql_data():
        mysql_data = database.get_records(config.labeled_data, columns=[config.data.target_column, args.target_column]) 
        if len(mysql_data) < 384:
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
def load_test_data(args, config, database, embedding_model):
    test_data = database.get_records(
        f'select {config.data.target_column}, {args.target_column} from sms_spam_cd.initial_data where day(datetime) = 22', 
        columns=[config.data.target_column, args.target_column]
    ).squeeze()
    
    if test_data.shape[0] > 0:
        y_test = test_data.pop(args.target_column).to_numpy().squeeze() 
        x_test = preprocess_data(embedding_model, test_data, target_column=config.data.target_column) 
        return x_test, y_test
    else:
        raise ValueError('Missing test data in sql table')

 
@task(name='Insert/Update MySQL', cache_policy=NO_CACHE)
def update_db(db, data): 
    """Update result of the model classification to MySQL"""
    # reset last_batch column
    db.run_statement('UPDATE sms_spam_cd.metadata_result SET last_batch = False')
    
    # save result to mysql
    db.save_to_mysql(data=data) 