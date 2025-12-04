import pandas as pd
import numpy as np 
import argparse
import mlflow 
import json

from sklearn.decomposition import PCA
from prefect import flow, get_run_logger
from kafka import KafkaConsumer

from src.ml.model_training import train_model, load_model 
from src.data_loader.preprocessing import get_normalized_messages  
from src.config_folder.config_loader import get_config
from src.utils.util import setup_environment, create_required_folder_file, update_metadata, generate_metadata
 

@flow(name='SMS_SPAM_DETECTION')
def main(args):  
    try:
        logger = get_run_logger()
        config = get_config() 
        create_required_folder_file(config) 
        
        db, embedding_model, vectorstore = setup_environment(config)
        # metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
        # update_metadata(config, metadata)
        
        consumer = KafkaConsumer(
            topics=[args.topic],
            bootstrap_servers=[args.kafka_uri],
            auto_offset_reset="earliest",
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info("Consumer started and listening")
        for msg in consumer:
            logger.info(f"Received data block at {msg.timestamp}")

            # load data in message queue
            messages_metadata = pd.DataFrame(msg.value)
            
            # get messages' id
            msg_id = messages_metadata.loc[:, 'id']
             
            # get faiss indexx
            faiss_index = vectorstore.get_index()
            
            # find embeddings in faiss index by ids
            embeddings = faiss_index.similarity_search()
            
            # dimensional reduction 
            pca = PCA(n_components=min(embeddings.shape[0], embeddings.shape[-1]))
            scaled_embeddings = pca.fit_transform(embeddings)
            
            # load model
            ml_model = load_model(config, db.get_cursor()) 
             
            # y_pred, confidence_score = ml_model.predict(embeddings), ml_model.predict_proba(embeddings).max(axis=1)
            
            # data_metadata = generate_metadata(data[config.data.metadata_column], y_pred, confidence_score, config.models.confidence_score_threshold)
            
            # MISSING: update metadata of faiss by ids
             
            
            update_metadata(config)
    except Exception as e:
        raise Exception(e)
    finally:
        db.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection with SGDClassifier')
    p.add_argument("--mlflow_uri", type=str, default='file:./mlruns', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument("--experiment", type=str, default='SMS SPAM DETECTION')
    p.add_argument("--model_id", type=str, default=None, help='specify trained model, else use new model') 
    p.add_argument("--kafka_uri", type=str, default='localhost:9092')
    p.add_argument("--topic", type=str, default='text_embedding')
    args = p.parse_args()
        
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)