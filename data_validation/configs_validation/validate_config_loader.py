from pydantic import BaseModel
 

class DatabaseConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    schema: str    


class EmbeddingConfig(BaseModel):
    model_name: str
    batch_size: int
    normalize_embeddings: bool
    show_progress: bool
    

class MLModelConfig(BaseModel):
    model_name: str
    
    
class VectorstoreConfig(BaseModel):
    directory: str


class ProjectConfig(BaseModel):
    mlflow_uri: str
    experiment_name: str
    label_column: str
    target_column: str
    number_of_uncertain: int
    threshold: float
    
    database: DatabaseConfig
    embedding: EmbeddingConfig
    ml_model: MLModelConfig
    vectorstore: VectorstoreConfig
    
    initial_data: str
    test_data: str
    labeled_data: str
    stratified_sampling: str