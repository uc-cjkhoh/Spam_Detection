from pydantic import BaseModel, ConfigDict
from langchain_huggingface import HuggingFaceEmbeddings

import pandas as pd


class VectorstoreConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    directory: str
    filename: str
    embedding: HuggingFaceEmbeddings
    
    
class LoadDataFrameInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data: pd.DataFrame
    save: bool
    
    
class LabelUncertainConfig(BaseModel):
    sentences: list