from typing import List

from docx2txt import docx2txt  # type: ignore[import-untyped]

from core.rag.document.document import Document
from core.rag.document_loaders.loader import FileLoader


class DocxLoader(FileLoader):
    def _load(self) -> List[Document]:
        return [Document(text=docx2txt.process(self))]
