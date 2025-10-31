import pandas as pd
import numpy as np

# ... missing unit testing before import custom libraries

from prefect import flow, task
from loader.config_loader import get_config
from testing.data_loader import Database
from testing.preprocessing import PreprocessPipeline
from testing.embedding import EmbeddingPipeline
from testing.vectorstore import VectorStore
# from testing.ml_pipeline import MLPipeline


@task
def load_data(config, query) -> pd.DataFrame:
    db = Database(config)  
    data = db.retrieve_by_query(query)[:100]
    data.columns = config.data.column_name
    db.close_connection()
    return data

@task
def preprocess(data: pd.Series) -> pd.Series: 
    preprocess = PreprocessPipeline()
    return preprocess.text_normalize(data)

@task
def text_embedding(model_name: str, data: pd.Series) -> np.ndarray:
    embeder = EmbeddingPipeline(model_name=model_name)
    return embeder.embed_message(data)

@task
def store_vectors(embeddings: np.ndarray):
    vectorstore = VectorStore(embeddings.shape[1])
    vectorstore.write(embeddings)
    return vectorstore

@flow
def main():
    module_config = get_config()
    
    model_name = module_config.models.text_embedding.model_name
    query = "select id, payload from sms_spam_cd.data_tdr_spam_filter limit 1000"
    
    
    data = load_data(module_config, query)
    
    # ... missing data quality check
    
    data['payload'] = preprocess(data['payload'])
    
    # ... missing data quality check
    
    embeddings = text_embedding(model_name, data['payload'])
    
    # ... missing data quality check
    
    vectorstore = store_vectors(embeddings)

    vectorstore.close()
    
if __name__ == '__main__':
    main()