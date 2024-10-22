from typing import Callable, List, Optional
import re

from auto_flow.core.llm.utils import get_tokenizer

DEFAULT_QUOTA = "“”‘’「『」』"


def split_by_sep(text: str,
                 sep_pattern: str | None = None,
                 seps: str | None = None,
                 quota=DEFAULT_QUOTA,
                 not_sep_in_quota: bool = False,
                 keep_sep: bool = True) -> List[str]:
    """
    When not_sep_in_quota = True,
        in order to skip separators inside paired quotes,
        use non-capturing group (?:...) to match non-separation points.
    Currently not distinguishing between left and right parentheses.
    """
    if not sep_pattern:
        assert seps
        sep_pattern = rf'(?:[{seps}])(?=(?:[^{quota}]*[{quota}][^{quota}]*[{quota}])*[^{quota}]*$)' \
            if not_sep_in_quota else rf'(?:[{seps}])'
    if keep_sep:
        sep_pattern = '(' + sep_pattern + ')'
    sentences = re.split(sep_pattern, text)
    ret = []
    for i in range(len(sentences)):
        s = sentences[i]
        if not s.strip() or re.match(sep_pattern, s):
            continue
        if keep_sep and i + 1 < len(sentences):
            s += sentences[i + 1]
        ret.append(s)
    return ret


def get_splitter_by_sep(sep_pattern: Optional[str] = None,
                        seps: str | None = None,
                        quota=DEFAULT_QUOTA,
                        not_sep_in_quota: bool = False,
                        keep_sep: bool = True) -> Callable[[str], List[str]]:
    return lambda text: split_by_sep(text, sep_pattern, seps, quota, not_sep_in_quota, keep_sep)


def get_split_by_char() -> Callable[[str], List[str]]:
    return lambda text: list(text)


def get_default_token_length_fn() -> Callable[[str], int]:
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")

    def token_length(text: str) -> int:
        return len(encoder.encode(text, allowed_special="all"))

    return token_length


def get_token_length_fn(model: str) -> Callable[[str], int]:
    tokenizer = get_tokenizer(model)
    return lambda text: len(tokenizer(text))
