import ast
import numpy as np
import mlflow
import mlflow.sklearn

from tqdm import tqdm
from sklearn.linear_model import SGDClassifier

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

  
@flow 
def load_model(config, cursor): 
    experiment = mlflow.search_experiments(filter_string = f"name='{config.mlflow_config.experiment_name}'")[0].experiment_id
    model_list = mlflow.search_logged_models(experiment_ids=[experiment], order_by=[{'field_name': 'creation_timestamp', 'ascending': False}])
    
    if len(model_list) > 0:
        latest_run_id = model_list.source_run_id.iloc[0] 
        model_uri = f"run:/{latest_run_id}/{config.mlflow_config.model_name}"
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        model = initialize_model(config, cursor)
        
    return model


@task(cache_policy=NO_CACHE)  
def blob_to_numpy(imported_data):
    embeddings = []
    labels = []
    for row in tqdm(imported_data):
        embedding = row[0].decode('utf-8')
        label = row[1]
        
        embeddings.append(ast.literal_eval(embedding))
        labels.append(label)
        
    return np.asarray(embeddings), np.asarray(labels)

 
@task(cache_policy=NO_CACHE)   
def initialize_model(config: dict, cursor):
    model = SGDClassifier(loss='log_loss', class_weight='balanced')  
    
    cursor.execute(config.initialize_model.query)
    imported_data = cursor.fetchall()
    
    embeddings, labels = blob_to_numpy(imported_data)
    
    model.fit(embeddings, labels)
    
    with mlflow.start_run(run_name='model_initialization'):
        mlflow.sklearn.log_model(
            sk_model=model, 
            name=config.mlflow_config.model_name, 
            input_example=embeddings[:10] 
        )
    
    return model


def train_model(model, x: np.ndarray, y: np.ndarray, metadata: str = None): 
    with mlflow.start_run(run_name='model_training'):
        mlflow.log_param('model_name', str(type(model).__name__)) 
        mlflow.log_params('model_params', model.get_params())
        mlflow.log_param('inputs_metadata', metadata) 
        
        model.partial_fit(x, y)
        
        mlflow.sklearn.log_model(model, name=f"model")


