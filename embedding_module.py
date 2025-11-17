import pandas as pd
import numpy as np 
import argparse
import mlflow 

from prefect import flow

from src.data_loader.preprocessing import get_normalized_messages  
from src.config_folder.config_loader import get_config
from src.utils.util import setup_environment, create_required_folder_file, update_metadata, generate_metadata
 

@flow(name='SMS_SPAM_DETECTION')
def main(args):  
    try:
        config = get_config() 
        
        create_required_folder_file(config) 
        
        db, embedding_model, vectorstore = setup_environment(config)    
        
        metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
        
        update_metadata(config, metadata)
        
        for i in range(len(metadata)):  
            data_query = config.data.query.format(*metadata.iloc[i]) 
            
            data = db.run_query(data_query, columns=config.data.column_name) 
            
            messages = get_normalized_messages(data, target_column=config.data.target_column) 
            
            embeddings = np.asarray(embedding_model.embed_documents(messages))
            
            data_metadata = generate_metadata(data[config.data.metadata_column], None, None, config.models.confidence_score_threshold)
             
            vectorstore.write_to_vectorstore(zip(messages, embeddings), embedding_model, data_metadata) 
            
            update_metadata(config)
    except Exception as e:
        raise Exception(e)
    finally:            
        db.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection with SGDClassifier')
    p.add_argument("--mlflow_uri", type=str, default='file:./mlruns', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument("--model_id", type=str, default=None, help='specify trained model, else us new model') 
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)