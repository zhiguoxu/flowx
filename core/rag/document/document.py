from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str | None = None
    text: str
    metadata: dict = Field(default_factory=dict)
