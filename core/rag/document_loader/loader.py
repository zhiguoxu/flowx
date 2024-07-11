from typing import List, Iterator

from core.rag.document.document import Document


class DocumentLoader:
    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        if type(self).load != DocumentLoader.load:
            return iter(self.load())
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )
