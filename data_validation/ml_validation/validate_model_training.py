from pydantic import BaseModel, ConfigDict
from typing import Any

import numpy as np


class ModelBoneStructureConfig(BaseModel):
    model_name: str
    model: Any
    
    
class EvaluateInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x_test: np.ndarray
    y_test: np.ndarray


class LGBMConfig(BaseModel):
    model_name: str
    

class TrainInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x: np.ndarray
    y: np.ndarray
    

class PredictInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x: np.ndarray
    
    
class PredictProbaInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x: np.ndarray