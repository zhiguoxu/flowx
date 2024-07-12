from pathlib import Path
from typing import List, Dict, Any, Sequence, Type

from pydantic import Field, BaseModel

from core.rag.document.document import Document, MetadataMapping
from core.rag.document_loaders.loader import DocumentLoader, FileLoader, AutoFileLoader
from core.rag.utils import OneOrMany
from core.utils.utils import filter_kwargs_by_pydantic


class DirLoader(BaseModel, DocumentLoader):
    file_or_dir: OneOrMany[str | Path]
    pattern: OneOrMany[str] = "*"
    exclude_pattern: OneOrMany[str] = ()
    recursive: bool = False
    encoding: str = "utf-8"
    metadata: MetadataMapping = Field(default_factory=dict)
    suffix_and_loaders: Dict[str, Type[FileLoader]] = Field(default_factory=dict)
    loader_kwargs: dict[str, Any] = Field(default_factory=dict)

    def load(self) -> List[Document]:
        docs = []
        for file in self.collect_input_files():
            kwargs = filter_kwargs_by_pydantic(AutoFileLoader, {**self.model_dump(), **self.loader_kwargs})
            file_loader = AutoFileLoader(file_path=file, **kwargs)
            docs.extend(file_loader.load())
        return docs

    def collect_input_files(self) -> List[Path]:
        input_file_or_dir_ = [Path(item) for item in to_list(self.file_or_dir)]
        pattern = to_list(self.pattern)
        exclude_pattern = to_list(self.exclude_pattern)

        input_files = []
        input_files_set = set()
        for file_or_dir in input_file_or_dir_:
            if not file_or_dir.exists():
                raise ValueError(f"path {file_or_dir} does not exist!")
            if file_or_dir.is_file():
                input_files.append(file_or_dir)
            else:
                path = file_or_dir
                exclude_files = set()
                for exclude_item in exclude_pattern:
                    for sub_file in path.rglob(exclude_item) if self.recursive else path.glob(exclude_item):
                        exclude_files.add(str(sub_file))

                for pattern_item in pattern:
                    for sub_file in path.rglob(pattern_item) if self.recursive else path.glob(pattern_item):
                        if not sub_file.is_file():
                            continue

                        sub_file_str = str(sub_file)
                        if sub_file_str in input_files_set:
                            continue

                        input_files_set.add(sub_file_str)

                        skip = False
                        for exclude_file in exclude_files:
                            if sub_file_str.startswith(exclude_file):
                                skip = True
                                break
                        if skip:
                            continue

                        input_files.append(sub_file)

        return input_files


def to_list(obj: Any) -> list:
    if isinstance(obj, str):
        return [obj]
    return list(obj) if isinstance(obj, Sequence) else [obj]
