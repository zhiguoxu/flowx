from pydantic import BaseModel

from core.rag.document_loaders.loader import DocumentLoader


class WikipediaLoader(BaseModel, DocumentLoader):
    ...
