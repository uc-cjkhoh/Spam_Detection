import pandas as pd

# ... missing unit testing before import custom libraries

from loader.config_loader import cfg
from testing.data_loader import get_connector
from testing.preprocessPipeline import PreprocessPipeline
from testing.embedding import EmbeddingPipeline

# Process
# 1. load data from mysql
# 2. preprocess message
# 3. perform embedding
# 4. save to vector database

def main():
    con = get_connector()
    cur = con.cursor()
    
    cur.execute("select id, payload from sms_spam_cd.data_tdr_spam_filter limit 1000")
    data = pd.DataFrame(cur.fetchall(), columns=['id', 'payload'])[:100]
    
    # ... missing data quality check
    
    preprocess = PreprocessPipeline()
    data['payload'] = preprocess.text_normalize(data['payload'])
    
    # ... missing data quality check
    
    embeder = EmbeddingPipeline(model_name=cfg.models.text_embedding.model_name)
    embeddings = embeder.embed_message(data['payload'])
    
    # ... missing data quality check
    
    
    
    print(embeddings)
    
    
if __name__ == '__main__':
    main()