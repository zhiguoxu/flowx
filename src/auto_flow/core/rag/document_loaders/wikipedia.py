from typing import Iterator

from pydantic import BaseModel

from auto_flow.core.rag.document.document import Document
from auto_flow.core.rag.document_loaders.loader import DocumentLoader

from auto_flow.core.rag.retrivers.wiki_retriever import WikiRetriever
from auto_flow.core.utils.utils import filter_kwargs_by_pydantic


class WikiLoader(BaseModel, DocumentLoader):
    query: str
    top_k: int = 3
    lang: str = "en"
    load_all_available_meta: bool = False
    max_query_length: int = 300
    doc_content_chars_max: int = 4000

    def lazy_load(self) -> Iterator[Document]:
        kwargs = filter_kwargs_by_pydantic(WikiRetriever, self.model_dump())
        wiki_retriever = WikiRetriever(**kwargs)
        yield from wiki_retriever.lazy_load(self.query)
