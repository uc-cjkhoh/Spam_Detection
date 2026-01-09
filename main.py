import numpy as np
import pandas as pd 
import argparse
import mlflow
import sys
import os
 
from prefect import flow, task 
from prefect.cache_policies import NO_CACHE
from sklearn.decomposition import PCA  
from imblearn.over_sampling import SMOTE

from src.data_loader.preprocessing import get_normalized_messages 
from src.utils.util import setup_core_components, update_metadata, finish_labelling
from src.ml.model_training import SGD, XGBoost
 

@task(name='Setup Environment', cache_policy=NO_CACHE) 
def setup_environment():  
    return setup_core_components()  

@task(name="Dimension Reduction", cache_policy=NO_CACHE)
def dimension_reduction(embeddings: np.ndarray):
    pca = PCA(n_components=384)
    scaled_embedding = pca.fit_transform(embeddings)
    return scaled_embedding

@task(name="Normalize Message", cache_policy=NO_CACHE)
def normalize_message(data, target_column):
    return get_normalized_messages(data, target_column)
     
@task(name="Sentence Embeddings", cache_policy=NO_CACHE)
def get_embeddings(normalized_message, embedding_model):    
    return np.asarray(embedding_model.embed_documents(normalized_message)) 
 
@task(name='Download First Batch Data', cache_policy=NO_CACHE)
def download_initial_data(config, db, filepath, target_column):
    data = db.run_query(config.data.query, columns=config.data.column_name) 
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
    y = data.loc[:, args.target_column].astype(int)
    
    resampled_x, resampled_y = oversampling(x, y) 
    model = model.fit(resampled_x, resampled_y)
  
    with mlflow.start_run(run_name='Build/Update Model'): 
        mlflow.log_param('embedding_model', config.models.text_embedding.model_name)
        mlflow.log_param('model_parameters', model.get_params())
        mlflow.sklearn.log_model(
            sk_model=model,
            name=type(model).__name__,
            registered_model_name=f'{type(model).__name__}',
            input_example=resampled_x[:1]
        )
 
@flow(name='Active Learning Pipeline')
def main(args):  
    try:
        # Setup necessary components
        config, database, _, embedding_model, _, models = setup_environment()  
        
        # Download first batch of stratified sample in local
        if not os.path.exists(config.models.initial_data_filepath): 
            download_initial_data(config, database, config.models.initial_data_filepath, args.target_column)
            sys.exit('Please label data manually before proceeding')
        
        # Check if finish labelling first batch of data
        if not finish_labelling(config.models.initial_data_filepath, args.target_column):
            raise ValueError(f"Please finish labelling the data in {config.models.initial_data_filepath}")
                      
        # Load target model classes
        xgboost = XGBoost(config.mlflow_config.experiment_name)
        sgd = SGD(config.mlflow_config.experiment_name)
        models = [sgd, xgboost]
        
        days = database.run_query('select distinct day(datetime) from sms_spam_cd.metadata_result', columns=['day'])['day']
        for day in days:
            evaluation = 0
            while evaluation < 0.8:
                for model in models:
                    # skip the first train_models when module start if model exists
                    skip_initialization = len(model.get_existing_models()) > 0
                    if not skip_initialization:
                        # build base model
                        train_models(config, args, database, embedding_model, model)
                        
                    # Select stratified sample in this day, group by hour
                    data = database.run_query(config.data.query, columns=config.data.column_name)
            
                    # Normalize message
                    normalized_message = normalize_message(data, target_column=config.data.target_column)
            
                    # Convert to vectors
                    _, embeddings = get_embeddings(normalized_message, embedding_model)
                    
                    # Dimension Reduction            
                    scaled_embeddings = dimension_reduction(embeddings)
            
                    # Classification
                    result, confidence_score = model.predict(scaled_embeddings), model.predict_proba(scaled_embeddings)
                    
                    # Label them by confidence score
                    high_conf_ids = np.where(confidence_score >= 0.975)[0]
                    uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), 1000)[:1000]
                    label_status = np.zeros(confidence_score.shape)
                    label_status[high_conf_ids] = 1
                    label_status[uncertain_ids] = -1
                    
                    # save result to mysql
                    database.save_to_mysql(
                        data=pd.DataFrame({
                            'id': data['id'],
                            'datetime': data['datetime'],
                            'spam_label': result,
                            'confidence_score': confidence_score,
                            'label_status': label_status,
                            'model': type(model).__name__
                        }).to_dict(orient='records')
                    )
                    
                    # update model
                    train_models(config, args, database, embedding_model, model)
                
                print('You can check result in MySQL and update labels')
                user_input = input('Press any key to process or `q` to quit ...')
                if user_input == 'q':
                    sys.exit('Module terminated')
                
                            
    except Exception as e:
        raise Exception(e)
    finally:            
        database.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument("--mlflow_uri", type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument('--target_column', type=str, default='spam_label')
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)