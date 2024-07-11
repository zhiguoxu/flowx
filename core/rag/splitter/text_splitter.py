import uuid
from abc import abstractmethod
from typing import List, Callable, Tuple, Any

from pydantic import BaseModel, model_validator, Field

from core.rag.document.document import Document, MetadataMode
from core.rag.splitter.utils import get_splitter_by_sep, get_split_by_char, get_default_token_length_fn

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_SENTENCE_SEPS = "。！？!?"
DEFAULT_SECONDARY_SEPS = ",，;；"
DEFAULT_TOKEN_SEPS = " "


class DocumentSplitter(BaseModel):
    chunk_size: int = DEFAULT_CHUNK_SIZE
    """The token chunk size for each chunk."""

    token_length_fn: Callable[[str], int] = Field(default_factory=get_default_token_length_fn)
    """"Calculate the token length of text, to satisfy the trunk size limit."""

    @abstractmethod
    def split_text(self, text: str, chunk_size: int | None = None) -> List[str]:
        ...

    def token_size(self, text: str) -> int:
        return self.token_length_fn(text)

    def split_document(self, documents: List[Document]) -> List[Document]:
        ret = []

        prev_doc: Document | None = None
        for doc in documents:
            meta_token_size = max(self.token_size(doc.get_metadata_str(mode=MetadataMode.EMBED)),
                                  self.token_size(doc.get_metadata_str(mode=MetadataMode.LLM)))
            for chunk_text in self.split_text(doc.text, self.chunk_size - meta_token_size):
                chunk_doc = doc.copy_without_rel()
                chunk_doc.id = str(uuid.uuid4())
                chunk_doc.text = chunk_text

                if prev_doc:
                    prev_doc.add_next(chunk_doc)

                ret.append(chunk_doc)
                prev_doc = chunk_doc

        return ret


class TextSplitter(DocumentSplitter):
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    """The token overlap of each chunk when splitting."""

    sentence_seps: str = DEFAULT_SENTENCE_SEPS
    """Sentences seperator."""

    secondary_seps: str = DEFAULT_SECONDARY_SEPS
    """Backup sentences seperator."""

    token_seps: str = DEFAULT_TOKEN_SEPS
    """Token seperator."""

    separate_fn_list: List[Callable]
    """Separate function list, it will be auto constructed if not specified by user."""

    @model_validator(mode='before')
    @classmethod
    def check_separate_fn_list(cls, data: Any) -> Any:
        if not data.get("separate_fn_list"):
            import nltk
            sentence_seps = data.get("sentence_seps", DEFAULT_SENTENCE_SEPS)
            secondary_seps = data.get("secondary_seps", DEFAULT_SECONDARY_SEPS)
            token_seps = data.get("token_seps", DEFAULT_TOKEN_SEPS)

            data["separate_fn_list"] = [
                get_splitter_by_sep("\n\n\n"),
                get_splitter_by_sep("\n\n"),
                get_splitter_by_sep("\n"),
                nltk.sent_tokenize,  # nltk don't support chinese
                get_splitter_by_sep(seps=sentence_seps, not_sep_in_quota=True),  # support chinese
                get_splitter_by_sep(seps=sentence_seps),
                get_splitter_by_sep(seps=secondary_seps),
                get_splitter_by_sep(seps=token_seps),
                get_split_by_char()
            ]
            return data

    def split_text(self, text: str, chunk_size: int | None = None) -> List[str]:
        chunk_size = chunk_size or self.chunk_size
        return self._split_text_dfs(text, 0, chunk_size)

    def _split_text_dfs(self, text: str, fn_index: int, chunk_size: int) -> List[str]:
        if self.token_size(text) <= chunk_size:
            return [text]

        ret = []
        for sub in self.separate_fn_list[fn_index](text):
            ret.extend(self._split_text_dfs(sub, fn_index + 1, chunk_size))

        return self._merge(ret, chunk_size)

    def _merge(self, splits: List[str], chunk_size: int) -> List[str]:
        ret = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        cur_chunk_size = 0

        for split in splits:
            split_size = self.token_size(split)
            assert split_size <= chunk_size
            if cur_chunk_size + split_size <= chunk_size:
                cur_chunk.append((split, split_size))
                cur_chunk_size += split_size
            else:
                ret.append("".join(text for text, _ in cur_chunk))
                # collect overlap
                new_chunk = [(split, split_size)]
                new_chunk_size = split_size
                for index in range(len(cur_chunk) - 1, -1, -1):
                    if new_chunk_size - split_size + cur_chunk[index][1] > self.chunk_overlap:
                        break
                    if new_chunk_size + cur_chunk[index][1] > chunk_size:
                        break
                    cur_chunk.insert(0, cur_chunk[index])
                    new_chunk_size += cur_chunk[index][1]
                cur_chunk = new_chunk
                cur_chunk_size = new_chunk_size

        ret.append("".join(text for text, _ in cur_chunk))
        return ret
