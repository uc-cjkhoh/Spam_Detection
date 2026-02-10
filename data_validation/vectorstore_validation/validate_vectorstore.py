from pydantic import BaseModel


class VectorstoreConfig(BaseModel):
    directory: str
    filename: str
      