from pathlib import Path
from typing import List

import pypdf

from auto_flow.core.rag.document.document import Document
from auto_flow.core.rag.document_loaders.loader import FileLoader


class PDFLoader(FileLoader):
    split_by_pages: bool = True

    def _load(self) -> List[Document]:

        with Path(self.file_path).open("rb") as fp:
            pdf = pypdf.PdfReader(fp)
            docs = []

            if self.split_by_pages:
                for index, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    label = pdf.page_labels[index]
                    docs.append(Document(text=text, metadata={"page_label": label}))
            else:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                docs.append(Document(text=text))

        return docs
