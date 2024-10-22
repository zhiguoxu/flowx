from __future__ import annotations

from openai import BaseModel


class TokenUsage(BaseModel):
    completion_tokens: int = 0
    """Number of tokens in the generated completion."""

    prompt_tokens: int = 0
    """Number of tokens in the prompt."""

    total_tokens: int = 0
    """Total number of tokens used in the request (prompt + completion)."""

    def __add__(self, other: TokenUsage) -> TokenUsage:
        usage = self.model_copy()
        usage.completion_tokens += other.completion_tokens
        usage.prompt_tokens += other.prompt_tokens
        usage.total_tokens += other.total_tokens
        return usage
