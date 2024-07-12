from abc import ABC
from typing import List

from core.flow.flow import Flow
from core.rag.document.document import Document


class Retriever(Flow[str, List[Document]], ABC):
    ...
