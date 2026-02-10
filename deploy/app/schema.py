import pandas as pd

from sqlalchemy.engine import Engine

from pydantic import BaseModel, ConfigDict


# ===================== main.py =====================
class InputData(BaseModel):
    id: list[int]
    payload: list[str]
    
    
class DataToMySQL(BaseModel):
    id: list[int]
    result: list[int]
    confidence_score: list[float]
    

# =============== simulate_request.py ===============
class RetrieveData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    engine: Engine
    query: str
     

class ClassifyRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    api_uri: str
    data: pd.DataFrame
    rate: int 