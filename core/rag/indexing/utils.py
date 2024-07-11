from __future__ import annotations

import hashlib
import json
import uuid
from typing import Iterable, Iterator, TypeVar, Any, Set

from pydantic import model_validator

from core.rag.document.document import Document

T = TypeVar("T")

NAMESPACE_UUID = uuid.UUID(int=1986)


def _hash_string_to_uuid(input_string: str) -> uuid.UUID:
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


class HashedDocument(Document):
    uid: str
    hash_: str
    """The hash of the document including content and metadata."""
    text_hash: str
    metadata_hash: str

    @model_validator(mode='before')
    @classmethod
    def check_card_number_omitted(cls, data: Any) -> Any:
        data["text_hash"] = str(_hash_string_to_uuid(data["text"]))
        metadata = json.dumps(data["metadata"], sort_keys=True)
        data["metadata_hash"] = str(_hash_string_to_uuid(metadata))
        data["hash_"] = str(_hash_string_to_uuid(data["text_hash"] + data["metadata_hash"]))
        if doc_id := data.get("id") is not None:
            uid = doc_id
        else:
            uid = data["hash_"]
        data["uid"] = uid
        return data

    def to_document(self) -> Document:
        return Document(id=self.id or self.uid, text=self.text, metadata=self.metadata)

    @classmethod
    def from_document(cls, document: Document) -> HashedDocument:
        return cls(id=document.id, text=document.text, metadata=document.metadata)


def dedup_docs(hashed_documents: Iterable[HashedDocument]) -> Iterator[HashedDocument]:
    seen: Set[str] = set()
    for hashed_doc in hashed_documents:
        if hashed_doc.hash_ not in seen:
            seen.add(hashed_doc.hash_)
            yield hashed_doc
