import pandas as pd
import numpy as np 
import argparse
import mlflow 

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

# custom libraries  
from src.data_loader.preprocessing import text_normalize
from src.config_folder.config_loader import get_config
from src.utils.util import create_required_folder_file 
from src.utils.util import setup_environment
 

@task(cache_policy=NO_CACHE)
def text_preprocessing(data: pd.DataFrame, embedding_model, target_column: str):
    data = text_normalize(data, target_column=target_column)
    message_list = data[target_column].to_list()
    
    embeddings = embedding_model.embed_documents(message_list)
    text_embedding_pair = zip(message_list, embeddings) 
    
    return np.asarray(embeddings), text_embedding_pair 


@flow(name='SMS_SPAM_DETECTION')
def main(args):
    try:
        # get environment config
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        
        config = get_config() 
        
        # create required directory
        create_required_folder_file(config)
        
        # setup environment
        mysql, embedding_model, vectorstore, ml_model = setup_environment(config, args)
        
        # get subdata's metadata
        metadata = mysql.get_population_metadata(config.metadata.query, columns=config.metadata.column_name)
        
        # start active learning process
        for i in range(len(metadata)):
            # 1. edit subdata's query
            data_query = config.data.query.format(*metadata.iloc[i])
             
            # 2. execute query & get subdata
            data, data_metadata = mysql.retrieve_subdata_by_query(data_query, columns=config.data.column_name)
            
            # 3. perform text normalization and embeddings
            embeddings, text_embedding_pair = text_preprocessing(data, embedding_model, target_column=config.data.target_column)
            
            # 5. write texts into vector database
            vectorstore.write_to_vectorstore(text_embedding_pair, embedding_model, data_metadata)
            
            # 6. store vectorbase to local storage
            vectorstore.save()
            
            # 7. classification
            labels = ml_model.classify_message(
                model=args.model_id,
                embeddings=embeddings,
                metadata=metadata
            )
            
            print(labels)
            # similar_text = loaded_index.similarity_search(query='you win a prize', k=5)
            
            break
                     
        mysql.close_connection() 
    except Exception as e:
        raise Exception(e) 
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection with SGDClassifier')
    p.add_argument("--mlflow_uri", type=str, default='file:./mlruns', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument("--model_id", type=str, default=None, help='specify trained model, else us new model')
    p.add_argument("--model_path", type=str, default=None, help='specify model to train')
    args = p.parse_args()
        
    main(args)