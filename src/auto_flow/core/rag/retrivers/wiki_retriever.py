from typing import List, Tuple, Iterator

from pydantic import model_validator
from typing_extensions import Self
from wikipedia import WikipediaPage  # type: ignore[import-untyped]
import wikipedia

from auto_flow.core.rag.document.document import Document
from auto_flow.core.rag.retrivers.retriever import Retriever


class WikiRetriever(Retriever):
    top_k: int = 3
    lang: str = "en"
    load_all_available_meta: bool = False
    max_query_length: int = 300
    doc_content_chars_max: int = 4000

    @model_validator(mode='after')
    def config_wikipedia(self) -> Self:
        wikipedia.set_lang(self.lang)
        return self

    def lazy_load(self, query: str) -> Iterator[Document]:
        page_titles = wikipedia.search(query[:self.max_query_length], results=self.top_k)
        for page_title in page_titles[: self.top_k]:
            if wiki_page := self._fetch_page(page_title):
                if doc := self._page_to_document(page_title, wiki_page):
                    yield doc

    def invoke(self, query: str) -> List[Document]:
        return list(self.lazy_load(query))

    def stream(self, query: str) -> Iterator[List[Document]] :
        for doc in self.lazy_load(query):
            yield [doc]

    def invoke_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    @classmethod
    def _fetch_page(cls, title: str) -> WikipediaPage | None:
        try:
            return wikipedia.page(title=title, auto_suggest=False)
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            return None

    def _page_to_document(self, page_title: str, wiki_page: WikipediaPage) -> Document:
        main_meta = {
            "title": page_title,
            "summary": wiki_page.summary,
            "source": wiki_page.url,
        }
        add_meta = (
            {
                "categories": wiki_page.categories,
                "page_url": wiki_page.url,
                "image_urls": wiki_page.images,
                "related_titles": wiki_page.links,
                "parent_id": wiki_page.parent_id,
                "references": wiki_page.references,
                "revision_id": wiki_page.revision_id,
                "sections": wiki_page.sections,
            }
            if self.load_all_available_meta
            else {}
        )
        doc = Document(
            text=wiki_page.content[: self.doc_content_chars_max],
            metadata={**main_meta, **add_meta}
        )
        return doc
