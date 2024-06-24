from typing import Dict, List

from pydantic import BaseModel, Field


class GenerationArgs(BaseModel):
    max_new_tokens: int = Field(
        default=512,
        description="Number of tokens the model can output when generating a response.",
    )

    temperature: float = Field(
        default=0.1,
        description="The temperature to use during generation.",
        ge=0.0,
    )

    stream: bool = Field(default=False, description="Streaming output.")

    repetition_penalty: float = 1

    stop: str | List[str] | None = None

    n: int = Field(default=1, description="How many chat completion choices to generate for each input message.")

    extra_kwargs: Dict | None = Field(default=None, description="Other generation parameters.")

    class Config:
        extra = "forbid"
