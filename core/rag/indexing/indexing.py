from typing import Sequence, Iterable, Set, Callable, cast, List

from pydantic import BaseModel

from core.rag.document.document import Document
from core.rag.document_loaders.loader import DocumentLoader
from core.rag.indexing.index_data_manager import IndexDataManager
from core.rag.indexing.utils import dedup_docs, HashedDocument
from core.rag.utils import batch
from core.rag.vectorstore.vectorstore import VectorStore


class IndexingResult(BaseModel):
    """Return a detailed a breakdown of the result of the indexing operation."""

    num_added: int
    """Number of added documents."""
    num_updated: int
    """Number of updated documents."""
    num_deleted: int
    """Number of deleted documents."""
    num_skipped: int
    """Number of skipped documents because they were already up to date."""


class Index(BaseModel):
    index_data_manager: IndexDataManager
    vector_store: VectorStore
    source_id_key: str | Callable[[Document], str] | None = None

    def add(self,
            docs_source: DocumentLoader | Iterable[Document],
            *,
            batch_size: int = 100,
            source_id_key: str | Callable[[Document], str] | None = None,
            clean_older_source: bool = False,
            force_update: bool = False) -> IndexingResult:
        source_id_key = source_id_key or self.source_id_key

        if isinstance(docs_source, DocumentLoader):
            try:
                doc_iterator = docs_source.lazy_load()
            except NotImplementedError:
                doc_iterator = iter(docs_source.load())
        else:
            doc_iterator = iter(docs_source)

        source_id_assigner = _get_source_id_assigner(source_id_key)

        index_start_time = self.index_data_manager.get_time()
        num_added = 0
        num_skipped = 0
        num_updated = 0
        num_deleted = 0

        for doc_batch in batch(batch_size, doc_iterator):
            hashed_docs = list(
                dedup_docs([HashedDocument.from_document(doc) for doc in doc_batch])
            )

            source_ids: Sequence[str | None] = [source_id_assigner(doc) for doc in hashed_docs]
            exists_batch = self.index_data_manager.exists([doc.id for doc in hashed_docs])

            # Filter out documents that already exist in the index manager.
            ids = []
            docs_to_index = []
            ids_to_skip = []
            seen_docs: Set[str] = set()
            for hashed_doc, doc_exists in zip(hashed_docs, exists_batch):
                if doc_exists:
                    if force_update:
                        seen_docs.add(hashed_doc.id)
                    else:
                        ids_to_skip.append(hashed_doc.id)
                        continue
                ids.append(hashed_doc.id)
                docs_to_index.append(hashed_doc.to_document())

            # 1. Update refresh timestamp
            if ids_to_skip:
                # Update skip documents or them will be deleted if clean_older_source = True.
                self.index_data_manager.update(ids_to_skip, index_start_time)
                num_skipped += len(ids_to_skip)

            # 2. Write to vector store
            if docs_to_index:
                self.vector_store.add_documents(docs_to_index, ids=ids, batch_size=batch_size)
                num_added += len(docs_to_index) - len(seen_docs)
                num_updated += len(seen_docs)

            # 3. Update index, even if they already exist since we want to refresh their timestamp.
            self.index_data_manager.update([doc.id for doc in hashed_docs], index_start_time, source_ids=source_ids)

            if clean_older_source:
                for source_id in source_ids:
                    if source_id is None:
                        raise AssertionError("Source ids cannot be None when clean_older_source = True.")

                _source_ids = cast(Sequence[str], source_ids)
                ids_to_delete = self.index_data_manager.list_keys(source_ids=_source_ids, before=index_start_time)
                if ids_to_delete:
                    # 1. delete from vector store.
                    self.vector_store.delete(ids_to_delete)
                    # 2. delete from index manager.
                    self.index_data_manager.delete(ids_to_delete)
                    num_deleted += len(ids_to_delete)

        return IndexingResult(num_added=num_added,
                              num_updated=num_updated,
                              num_skipped=num_skipped,
                              num_deleted=num_deleted)

    def delete(self, ids: List[str]) -> None:
        self.vector_store.delete(ids)
        self.index_data_manager.delete(ids)

    def delete_by_source(self, source_id: str) -> List[str]:
        ids = self.index_data_manager.list_keys(source_ids=[source_id])
        self.delete(ids)
        return ids

    def delete_all(self) -> None:
        self.vector_store.delete_all()
        self.index_data_manager.delete_all()

    class Config:
        arbitrary_types_allowed = True


def _get_source_id_assigner(source_id_key: str | Callable[[Document], str] | None
                            ) -> Callable[[Document], str | None]:
    """Get the source id from the document."""
    if source_id_key is None:
        return lambda doc: None
    elif isinstance(source_id_key, str):
        return lambda doc: str(doc.metadata[source_id_key])
    elif callable(source_id_key):
        return source_id_key
    else:
        raise ValueError(
            f"source_id_key should be either None, a string or a callable. "
            f"Got {source_id_key} of type {type(source_id_key)}."
        )
