from functools import partial
from typing import List, Callable

from auto_flow.core.logging import get_logger

logger = get_logger(__name__)


# Tokenizer for calculate encoding length.
def get_tokenizer(model: str) -> Callable[[str], List[int]]:
    import tiktoken
    try:
        # try openai
        return partial(tiktoken.encoding_for_model(model).encode, allowed_special="all")
    except KeyError as e:
        ...
    try:
        # try huggingface
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        hf_tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return partial(hf_tokenizer.encode, add_special_tokens=False)
    except Exception as e:
        ...

    logger.warning("Warning: model not found. Using cl100k_base encoding.")
    return partial(tiktoken.get_encoding("cl100k_base").encode, allowed_special="all")
