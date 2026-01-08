import numpy as np
import pandas as pd 
import argparse
import mlflow
import sys
import os
 
from prefect import flow, task 
from prefect.cache_policies import NO_CACHE
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.data_loader.preprocessing import get_normalized_messages  
from src.config_folder.config_loader import get_config
from src.utils.util import setup_core_instances, create_required_folder_file, update_metadata, finish_labelling
from src.ml.model_training import SGD, XGBoost
 
 
@task(name='Setup Environment', cache_policy=NO_CACHE) 
def setup_environment():
    config = get_config() 
    create_required_folder_file(config) 
    db, embedding_model, faiss = setup_core_instances(config)
    metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
    update_metadata(config, metadata)
    return config, db, embedding_model, faiss, metadata

 
@task(name="Overlap with Least Confidence Embeddings", cache_policy=NO_CACHE)
def overlapping_embeddings(db, faiss, new_batch_mtds, new_batch_embeddings, max_n = 5000):  
    if faiss and faiss.index:
        query = f"SELECT row_id, id, datetime FROM sms_spam_cd.metadata_result WHERE label_status = 'psuedo_label' ORDER BY confidence_score ASC LIMIT {max_n}"
        metadata = db.run_query(query, columns=['row_id', 'id', 'datetime']) 
        least_confidence_embeddings = faiss.index.index.reconstruct_batch(metadata['row_id'].tolist())
        
        combined_mtds = pd.concat([metadata, new_batch_mtds])
        combined_embeddings = np.concatenate([least_confidence_embeddings, new_batch_embeddings])
        return combined_mtds, combined_embeddings
    else:
        return new_batch_mtds, new_batch_embeddings
   
       
@task(name='Sentence Embeddings', cache_policy=NO_CACHE)
def embed_messages(embedding_model, messages: list):
    return np.asarray(embedding_model.embed_documents(messages))


@task(name="Dimension Reduction", cache_policy=NO_CACHE)
def dimension_reduction(embeddings: np.ndarray):
    pca = PCA(n_components=384)
    scaled_embedding = pca.fit_transform(embeddings)
    return scaled_embedding


@task(name="Data Preprocessing", cache_policy=NO_CACHE)
def get_embeddings(config, data, embedding_model):   
    data = data.copy()
    data = data.fillna('')
    
    messages = get_normalized_messages(data, target_column=config.data.target_column)
    embeddings = embed_messages(embedding_model, messages)    
    return messages, embeddings
 

@task(name='Download First Batch Data', cache_policy=NO_CACHE)
def download_initial_data(config, db, filepath, target_column):
    data = db.run_query(config.data.query.format(*("16, 17, 18, 19, 20, 21, 22", 300)), columns=config.data.column_name) 
    data[target_column] = None
    
    data.to_excel(filepath, index=False) 

  
@task(name='Spam Classification', cache_policy=NO_CACHE)
def spam_classification(model, embeddings): 
    return model.predict(embeddings), model.predict_proba(embeddings).max(axis=1)


@task(name='Load Data', cache_policy=NO_CACHE)
def load_data(db, config):
    initial_data = pd.read_excel(config.models.initial_data_filepath)
    pseudo_data =  db.run_query(config.pseudo_query, columns=initial_data.columns)
    human_data = db.run_query(config.human_query, columns=initial_data.columns)
    train_data = pd.concat([initial_data, pseudo_data, human_data])
    return train_data


@task(name='Oversampling', cache_policy=NO_CACHE)
def oversampling(x, y):
    smote = SMOTE(random_state=42)
    resampled_x, resampled_y = smote.fit_resample(x, y)    
    return resampled_x, resampled_y 
    

@task(name='Model Training', cache_policy=NO_CACHE)
def train_models(config, args, db, embedding_model, model):
    data = load_data(db, config)
     
    _, embeddings = get_embeddings(config, data, embedding_model)
    
    x = dimension_reduction(embeddings)
    y = data.loc[:, args.target_column]
    
    resampled_x, resampled_y = oversampling(x, y)

    if type(model).__name__ == 'XGBClassifier':
        x_train, x_test, y_train, y_test = train_test_split(resampled_x, resampled_y, test_size=0.2, shuffle=True)  
        model = model.fit(x_train, y_train, eval_set=[(x_test, y_test)], xgb_model=model.get_booster()) 
    elif type(model).__name__ == 'SGDClassifier':
        # require loop to perform partial fit 
        model = model.fit(resampled_x, resampled_y)
    
    # save model
    with mlflow.start_run(run_name='Initialize Model'): 
        mlflow.log_param('embedding_model', config.models.text_embedding.model_name)
        mlflow.log_param('model_parameters', model.get_params())
        mlflow.sklearn.log_model(
            sk_model=model,
            name=type(model).__name__,
            registered_model_name=f'{type(model).__name__}_Model',
            input_example=resampled_x[:1]
        )
     
    return model

     
@flow(name='Active Learning Pipeline')
def main(args):  
    try:
        # 1. Setup necessary components
        config, db, embedding_model, faiss, metadata = setup_environment()  
        
        # 2. Download first batch of stratified sample in local
        if not os.path.exists(config.models.initial_data_filepath): 
            download_initial_data(config, db, config.models.initial_data_filepath, args.target_column)
            sys.exit('Please label data manually before proceeding')
        
        # 3. Check if finish labelling first batch of data
        if not finish_labelling(config.models.initial_data_filepath, args.target_column):
            raise ValueError(f"Please finish labelling the data in {config.models.initial_data_filepath}")
                      
        # 4. Load target model classes
        xgboost = XGBoost(config.mlflow_config.experiment_name)
        sgd = SGD(config.mlflow_config.experiment_name)
        model_classes = [sgd, xgboost]
        
        days = db.run_query('select distinct day(current_datetime) from sms_spam_cd.metadata_result', columns=['day'])['day']
        for day in days:
            evaluation = 0
            while evaluation < 0.8:  
                for model_class in model_classes:
                    # 5. Initiate or train model
                    model = train_models(config, args, db, embedding_model, model_class.model)
                    
                    # 6. Select stratified sample in this day, group by hour
                    data = db.run_query(config.data.query.format(*(day, 2000)), columns=config.data.column_name)
            
                    # 7. Convert to vectors
                    _, embeddings = get_embeddings(config, data, embedding_model)
                    
                    # 8. Dimension Reduction            
                    scaled_embeddings = dimension_reduction(embeddings)
            
                    # 9. Classification
                    result, confidence_score = model.predict(scaled_embeddings), model.predict_proba(scaled_embeddings)
                    
                    # 10. Label them by confidence score
                    high_conf_ids = np.where(confidence_score >= 0.975)[0]
                    uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), 1000)[:1000]
                    
                    label_status = np.zeros(confidence_score.shape)
                    label_status[high_conf_ids] = 1
                    label_status[uncertain_ids] = -1
                    
                    db.save_to_mysql(
                        data=pd.DataFrame({
                            'id': data['id'],  
                            'datetime': data['datetime'],
                            'spam_label': result,
                            'confidence_score': confidence_score,
                            'label_status': label_status
                        }).to_dict(orient='records')
                    )
                            
    except Exception as e:
        raise Exception(e)
    finally:            
        db.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument("--mlflow_uri", type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument('--target_column', type=str, default='spam_label')
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)