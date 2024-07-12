from pathlib import Path
from typing import List, Iterator, Type, Dict, Any

from pydantic import Field, model_validator, BaseModel, field_validator
from typing_extensions import Self

from core.logging import get_logger
from core.rag.document.document import Document, MetadataMapping
from core.utils.utils import filter_kwargs_by_pydantic

logger = get_logger(__name__)


class DocumentLoader:
    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        if type(self).load != DocumentLoader.load:
            return iter(self.load())
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()"
        )


class FileLoader(BaseModel, DocumentLoader):
    file_path: str | Path
    metadata: MetadataMapping = Field(default_factory=dict)
    encoding: str = "utf-8"

    @model_validator(mode="after")
    def set_metadata(self) -> Self:
        self.metadata = {**self.metadata, "source": str(self.file_path)}
        return self

    def load(self) -> List[Document]:
        logger.warning(f"Use default loader for {self.file_path}")
        text = Path(self.file_path).read_text(encoding=self.encoding, errors="ignore")
        return [Document(text=text, metadata=self.metadata)]


class AutoFileLoader(FileLoader):
    suffix_and_loaders: Dict[str, Type[FileLoader]] = Field(default_factory=dict)
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def set_suffix_and_loaders(self) -> Self:
        self.suffix_and_loaders = {**default_suffix_file_loaders(), **self.suffix_and_loaders}
        return self

    def load(self) -> List[Document]:
        suffix = Path(self.file_path).suffix.lower()[1:]
        loader_type = self.suffix_and_loaders.get(suffix, FileLoader)
        kwargs = filter_kwargs_by_pydantic(loader_type, {**self.model_dump(), **self.loader_kwargs})
        return loader_type(**kwargs).load()


def default_suffix_file_loaders() -> Dict[str, Type[FileLoader]]:
    from core.rag.document_loaders.file_loaders.pdf_loader import PDFLoader
    from core.rag.document_loaders.file_loaders.docx_loader import DocxLoader
    return dict(
        pdf=PDFLoader,
        docx=DocxLoader
    )
