import numpy as np
import os
import pandas as pd
import argparse
import json

from tqdm import tqdm
from prefect import flow, task
from kafka import KafkaProducer

from src.data_loader.preprocessing import get_normalized_messages  
from src.config_folder.config_loader import get_config
from src.utils.util import setup_environment, create_required_folder_file, update_metadata, initialize_metadata
 
 
@task
def embed_messages(embedding_model, messages: list):
    return np.asarray(embedding_model.embed_documents(messages))

@task
def send_to_message_queue(producer: KafkaProducer, topic: str, embeddings: np.ndarray, metadatas: pd.DataFrame, batch_size: int):
    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        payload = {
            'batch_idx': i,
            'embeddings': batch_embeddings,
            'metadatas': batch_metadatas
        }
        producer.send(topic=topic, value=payload)
        producer.flush()
    

@flow(name='Text Embedding')
def main(args):  
    try:
        config = get_config() 
        create_required_folder_file(config) 
        
        db, embedding_model, vectorstore = setup_environment(config)
        metadata = db.run_query(config.metadata.query, columns=config.metadata.column_name) 
        
        update_metadata(config, metadata)
        
        producer = KafkaProducer(
            bootstrap_servers=[args.kafka_uri],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        for i in range(len(metadata)):  
            # get messages
            data_query = config.data.query.format(*metadata.iloc[i]) 
            data = db.run_query(data_query, columns=config.data.column_name) 
            messages = get_normalized_messages(data, target_column=config.data.target_column) 
            
            # save messages
            embeddings = embed_messages(embedding_model, messages)
            data_metadata = initialize_metadata(data[config.data.metedata_column])
            vectorstore.write_to_vectorstore(zip(messages, embeddings), embedding_model, data_metadata) 
            
            # send messages
            send_to_message_queue(
                producer=producer, 
                topic=args.topic, 
                embeddings=embeddings,
                metadatas=data_metadata,
                batch_size=args.batch_size
            )
            
            break
            
            update_metadata(config)
    except Exception as e:
        raise Exception(e)
    finally:            
        db.close_connection()  
    
    
if __name__ == '__main__': 
    p = argparse.ArgumentParser(description="Text Embedding Module")
    p.add_argument("--kafka_uri", type=str, default='localhost:9092')
    p.add_argument("--topic", type=str, default='text_embedding')
    p.add_argument("--batch_size", type=int, default=500)
    
    args = p.parse_args()
    main(args)