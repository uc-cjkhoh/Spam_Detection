from pydantic import BaseModel, ConfigDict 
from data_validation.configs_validation.validate_config_loader import DatabaseConfig


class MySQLConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    db_config: DatabaseConfig


class RunStatement(BaseModel):
    statement: str
    
    
class GetRecords(BaseModel):
    query: str
    

class SaveToMySQL(BaseModel):
    data: dict