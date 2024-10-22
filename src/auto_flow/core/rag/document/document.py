from __future__ import annotations

import uuid
from enum import Enum, auto
from typing import List, Dict, Mapping

from pydantic import BaseModel, Field

MetadataValueType = str | int | float | bool
MetadataMapping = Mapping[str, MetadataValueType]


class MetadataMode(str, Enum):
    ALL = "all"
    EMBED = "embed"
    LLM = "llm"
    NONE = "none"


class DocRelationType(str, Enum):
    PARENT = auto()
    CHILD = auto()
    NEXT = auto()
    PREV = auto()


class RelatedDocInfo(BaseModel):
    doc_id: str
    metadata: MetadataMapping


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: MetadataMapping = Field(default_factory=dict)

    excluded_embed_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model."
    )

    excluded_llm_metadata_keys: List[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM."
    )

    text_template: str = Field(
        default="{metadata_str}\n\n{content}",
        description=(
            "Template for how text is formatted, with {content} and {metadata_str} placeholders."
        )
    )

    metadata_template: str = Field(
        default="{key}: {value}",
        description=(
            "Template for how metadata is formatted, with {key} and {value} placeholders."
        )
    )

    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string."
    )

    relationships: Dict[DocRelationType, List[RelatedDocInfo]] = Field(
        default_factory=dict,
        description="A mapping of relationships to other document."
    )

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text

        return self.text_template.format(
            content=self.text, metadata_str=metadata_str
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join([
            self.metadata_template.format(key=key, value=str(value))
            for key, value in self.metadata.items()
            if key in usable_metadata_keys
        ])

    def as_related_doc_info(self) -> RelatedDocInfo:
        return RelatedDocInfo(doc_id=self.id, metadata=self.metadata)

    @property
    def parent(self) -> RelatedDocInfo | None:
        if DocRelationType.PARENT not in self.relationships:
            return None

        return self.relationships[DocRelationType.PARENT][0]

    @property
    def children(self) -> List[RelatedDocInfo]:
        if DocRelationType.CHILD not in self.relationships:
            return []

        return self.relationships[DocRelationType.CHILD]

    @property
    def next(self) -> RelatedDocInfo | None:
        if DocRelationType.NEXT not in self.relationships:
            return None

        return self.relationships[DocRelationType.NEXT][0]

    @property
    def prev(self) -> RelatedDocInfo | None:
        if DocRelationType.PREV not in self.relationships:
            return None

        return self.relationships[DocRelationType.PREV][0]

    def add_next(self, next_doc: Document):
        self.relationships[DocRelationType.NEXT] = [next_doc.as_related_doc_info()]
        next_doc.relationships[DocRelationType.PREV] = [self.as_related_doc_info()]

    def add_child(self, child: Document):
        child.relationships[DocRelationType.PARENT] = [self.as_related_doc_info()]
        child_docs = self.children or []
        child_docs.append(child.as_related_doc_info())
        self.relationships[DocRelationType.CHILD] = child_docs

    def copy_without_rel(self) -> Document:
        doc = self.model_copy(deep=True)
        doc.relationships = {}
        return doc
