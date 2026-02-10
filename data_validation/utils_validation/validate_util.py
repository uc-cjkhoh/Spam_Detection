from pydantic import BaseModel, ConfigDict

from data_validation.configs_validation.validate_config_loader import ProjectConfig
from src.data_loader.database import Database

import numpy as np
import pandas as pd


class CreateFolderFilesConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: ProjectConfig
    db: Database
    

class SamplingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: ProjectConfig
    db: Database


class OversamplingInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x: pd.Series
    y: pd.Series
    k_neighbors: int
    
    
class GetUniquePatternInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    embeddings: np.ndarray