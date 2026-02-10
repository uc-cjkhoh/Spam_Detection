from pydantic import BaseModel
from typing import Any


class ModelBoneStructureConfig(BaseModel):
    model_name: str
    model: Any