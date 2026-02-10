from pydantic import BaseModel, ConfigDict

import pandas as pd


class FeatureEngineering(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: pd.Series
    
    
class CleanText(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: pd.Series