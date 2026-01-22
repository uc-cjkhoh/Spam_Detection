import os
import sys
import mlflow
import argparse
import numpy as np
import pandas as pd 

from prefect import flow, task
from prefect.cache_policies import NO_CACHE 
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from imblearn.over_sampling import SMOTE
from langchain.schema import Document

from src.data_loader.preprocessing import get_normalized_messages 
from src.utils.util import setup_core_components, save_evaluation
 
 
@task(name='Setup environment', cache_policy=NO_CACHE) 
def setup_environment(args):  
    config, database, embedding_model, vectorstore, teacher, student = setup_core_components(args)
    
    # check initial data status
    initial_data = database.get_records(
        f'select {config.data.target_column}, {args.target_column} from sms_spam_cd.initial_data where day(datetime) != 22', 
        columns=[config.data.target_column, args.target_column]
    ).squeeze()
      
    if initial_data[args.target_column].isna().sum():  
        sys.exit(f"Please finish labelling the data in table `sms_spam_cd.initial_data`")

    # check if vectorstore created
    if len(os.listdir(vectorstore.directory)) > 0:
        vectorstore.load_index(folder_path=vectorstore.directory, index_name=vectorstore.filename)
    else:
        labels = initial_data.pop(args.target_column)
        features = get_normalized_messages(initial_data, config.data.target_column)
        payloads = features.pop(config.data.target_column)
        
        features.to_csv('./data/initial_data_features.csv', index=False)
        
        documents = [
            Document(
                page_content=payload,
                metadata={'label': label}
            )
            for payload, label in zip(payloads, labels)
        ] 
        vectorstore.write_index(documents)
        
    return config, database, embedding_model, vectorstore, teacher, student

@task(name='Spam classification', cache_policy=NO_CACHE)
def spam_classification(model, embeddings): 
    return model.predict(embeddings), model.predict_proba(embeddings)

@task(name='Model training', cache_policy=NO_CACHE)
def train_models(model, x, y):
    model_instance = model.fit(x, y) 
    with mlflow.start_run(run_name='Build/Update Model'):
        mlflow.log_param('model_parameters', model_instance.get_params())
        mlflow.sklearn.log_model(
            sk_model=model_instance,
            name=model.model_name,
            registered_model_name=f'{model.model_name}',
            input_example=x[:1]
        )
    return model

@task(name='Model Evaluation', cache_policy=NO_CACHE)
def evaluate_model(model, x, y_true):
    y_pred = model.predict(x)
    confidence_score = model.predict_proba(x)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, confidence_score)
    
    result = pd.DataFrame({
        'Model': model.model_name,
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'Loss': [loss]
    })
    
    save_evaluation(result) 

@task(name='Stratified Sampling', cache_policy=NO_CACHE)
def stratified_sampling(config, db):
    data = db.get_records(config.data.query, columns=config.data.column_name)
    return data['id'], data['datetime'], data[['payload']]
    
@task(name='Insert/Update MySQL', cache_policy=NO_CACHE)
def update_db(db, data): 
    # reset last_batch column
    db.run_statement('UPDATE sms_spam_cd.metadata_result SET last_batch = False')
    
    # save result to mysql
    db.save_to_mysql(data=data)

@task(name="Normalize message", cache_policy=NO_CACHE)
def normalize_message(data, target_column):
    return get_normalized_messages(data, target_column)
   
@task(name="Sentence embeddings", cache_policy=NO_CACHE)
def sentence_embeddings(messages: list, embedding_model):    
    return np.asarray(embedding_model.embed_documents(messages)) 
    
@task(name="Dimension reduction", cache_policy=NO_CACHE)
def dimension_reduction(embeddings: np.ndarray):
    pca = PCA(n_components=384)
    scaled_embedding = pca.fit_transform(embeddings)
    return scaled_embedding

@task(name='Data Preprocessing', cache_policy=NO_CACHE)
def preprocess_data(embedding_model, sms_message: pd.DataFrame, target_column: str) -> np.ndarray:  
    features = normalize_message(sms_message, target_column=target_column) 
    
    messages = features.pop(target_column)
    embeddings = sentence_embeddings(messages.to_list(), embedding_model)             
    scaled_embeddings = dimension_reduction(embeddings)
    
    scaler = RobustScaler()
    features[['feature_emoji_count', 'feature_length', 'feature_newline_count']] = scaler.fit_transform(
        features[['feature_emoji_count', 'feature_length', 'feature_newline_count']]
    )
    
    scaled_embeddings = np.hstack((features.to_numpy(), scaled_embeddings))
    return scaled_embeddings 

@task(name='Oversampling', cache_policy=NO_CACHE)
def oversampling(x, y):
    smote = SMOTE(random_state=42)
    resampled_x, resampled_y = smote.fit_resample(x, y)    
    return resampled_x, resampled_y

@task(name='Load train data', cache_policy=NO_CACHE)
def load_train_data(config, database, embedding_model, vectorstore): 
    @task(name='Load initial data', cache_policy=NO_CACHE)
    def load_initial_data():
        initial_embeddings = vectorstore.faiss.index.reconstruct_n(0, -1)  
        scaled_initial_embeddings = dimension_reduction(initial_embeddings)
        
        initial_features = pd.read_csv('./data/initial_data_features.csv')
        scaler = RobustScaler()
        initial_features[['feature_emoji_count', 'feature_length', 'feature_newline_count']] = scaler.fit_transform(
            initial_features[['feature_emoji_count', 'feature_length', 'feature_newline_count']]
        )
        
        scaled_embeddings = np.hstack((initial_features.to_numpy(), scaled_initial_embeddings))

        docs = list(vectorstore.faiss.docstore._dict.values())    
        initial_labels = [list(doc.metadata.values()) for doc in docs]
        initial_labels = np.asarray(initial_labels).squeeze()
   
        return scaled_embeddings, initial_labels
   
    @task(name='Load MySQL data', cache_policy=NO_CACHE)
    def load_mysql_data():
        mysql_data = database.get_records(config.labeled_data, columns=[config.data.target_column, args.target_column]) 
        if len(mysql_data) == 0:
            return np.array([]), np.array([])
            
        sql_labels = mysql_data.pop(args.target_column)
        sql_embeddings = preprocess_data(
            embedding_model=embedding_model,
            sms_message=mysql_data, 
            target_column=config.data.target_column
        )
        
        return sql_embeddings, sql_labels
   
    initial_embeddings, initial_labels = load_initial_data()
    sql_embeddings, sql_labels = load_mysql_data()

    if len(sql_embeddings) == 0: 
        x, y = oversampling(initial_embeddings, initial_labels)
        return x, y
    else:
        # combine initial data with sql data
        complete_train_embeddings = np.vstack((initial_embeddings, sql_embeddings)) 
        complete_train_labels = np.hstack((initial_labels, sql_labels)).reshape(-1, 1)
        x, y = oversampling(complete_train_embeddings, complete_train_labels)
        return x, y

@task(name='Load test data', cache_policy=NO_CACHE)
def load_test_data(config, database, embedding_model):
    test_data = database.get_records(
        f'select {config.data.target_column}, {args.target_column} from sms_spam_cd.initial_data where day(datetime) = 22', 
        columns=[config.data.target_column, args.target_column]
    ).squeeze()
    
    if test_data.shape[0] > 0:
        label = test_data.pop(args.target_column).to_numpy().squeeze() 
        embeddings = preprocess_data(embedding_model, test_data, target_column=config.data.target_column) 
        return embeddings, label
    else:
        raise ValueError('Missing test data in sql table')

@flow(name='Active Learning Pipeline')
def main(args):   
    try:
        config, database, embedding_model, vectorstore, teacher, student = setup_environment(args)   
        
        # load train data
        x_train, y_train = load_train_data(config, database, embedding_model, vectorstore)
        x_test, y_test = load_test_data(config, database, embedding_model)
         
        # initial teacher model if not skip
        if not args.skip_first_training:
            teacher = train_models(teacher, x_train, y_train)
        
        # evaluate teacher model
        evaluate_model(teacher, x_test, y_test)
        
        # stratified sampling
        data_id, data_dt, data_msg = stratified_sampling(config, database)

        # preprocess data
        scaled_embeddings = preprocess_data(embedding_model, data_msg, target_column=config.data.target_column)
        
        # classification
        result, confidence_score = spam_classification(teacher, scaled_embeddings)
        
        # prepare data to be store in mysql
        high_conf_ids = np.where(confidence_score >= args.threshold)[0]
        uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), args.number_of_uncertain)[:args.number_of_uncertain]
        label_status = np.zeros(confidence_score.shape)
        label_status[high_conf_ids] = 1
        label_status[uncertain_ids] = -1 
        
        high_conf_embeddings, high_conf_labels = scaled_embeddings[high_conf_ids], result[high_conf_ids].squeeze()
        pseud_x, pseud_y = np.vstack((x_train, high_conf_embeddings)), np.hstack((y_train, high_conf_labels))
        pseud_x, pseud_y = oversampling(pseud_x, pseud_y)
        
        # train student model 
        student = train_models(student, pseud_x, pseud_y.reshape(-1, 1))
        
        # evaluate student model
        evaluate_model(student, x_test, y_test)
         
        # save uncertainty from teacher model
        data_to_sql = pd.DataFrame({
            'id': data_id,
            'datetime': data_dt,
            'spam_label': result,
            'confidence_score': confidence_score,
            'label_status': label_status,
            'model': type(teacher).__name__,
            'last_batch': [1] * len(data_id)
        }).to_dict(orient='records')
        
        # update mysqlw
        update_db(database, data_to_sql)
         
    except Exception as e:
        raise Exception(e)
    finally:
        database.close_connection() 
               
               
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument('-u', '--mlflow_uri', type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument('-e', '--experiment', type=str, default='SMS_SPAM_DETECTION_V3', help='name of the experiment in mlflow')
    p.add_argument('-c', '--target_column', type=str, default='spam_label', help='the column in database that indicate the type of sms (spam or ham)')
    p.add_argument('-s', '--skip_initialization', type=bool, default=True, help='whether to skip model initialization')
    p.add_argument('-x', '--skip_first_training', type=int, default=0, help='whether to skip the first training'),
    p.add_argument('-n', '--number_of_uncertain', type=int, default=100, help='configure the number of uncertain message for human label')
    p.add_argument('-t', '--threshold', type=float, default=0.975, help='configure the confidence score threshold')
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)