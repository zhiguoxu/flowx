import json
from typing import Dict, Callable, List, Tuple, Any

import numpy as np
from chromadb import Settings, Where, WhereDocument
from chromadb.api.client import Client
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field

from auto_flow.core.rag.document.document import Document, MetadataMode
from auto_flow.core.rag.embeddings.embeddings import Embeddings
from auto_flow.core.rag.embeddings.huggingface.hf_embedding import HuggingfaceEmbeddings
from auto_flow.core.rag.utils import batch2, repeat
from auto_flow.core.rag.vectorstore.utils import mmr_top_k, calc_similarity
from auto_flow.core.rag.vectorstore.vectorstore import VectorStore
from auto_flow.core.utils.utils import filter_kwargs_by_pydantic

DEFAULT_COLLECTION_NAME = "flowx"


class Chroma(BaseModel, VectorStore):
    client: Client
    collection: Collection
    embedding_function: Embeddings = Field(default_factory=HuggingfaceEmbeddings)
    similarity_fn: Callable[..., np.ndarray] = calc_similarity

    def __init__(self,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 persist_dir: str | None = None,
                 client_settings: Settings | None = None,
                 collection_metadata: Dict | None = None,
                 client: Client | None = None,
                 embedding_function: Embeddings = HuggingfaceEmbeddings(),
                 similarity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = calc_similarity):

        if client is None:
            if client_settings:
                client_settings.persist_directory = persist_dir or client_settings.persist_directory
            elif persist_dir:
                client_settings = Settings(is_persistent=True, persist_directory=persist_dir)
            else:
                client_settings = Settings()
            client = Client(settings=client_settings)
        collection = client.get_or_create_collection(name=collection_name,
                                                     metadata=collection_metadata,
                                                     embedding_function=None)
        kwargs = filter_kwargs_by_pydantic(Chroma, locals(), exclude_none=True)
        super().__init__(**kwargs)

    def add_documents(self,
                      documents: List[Document],
                      ids: List[str] | None = None,
                      batch_size: int = 1_000) -> List[str]:
        ret_ids = []
        for docs_, ids_ in batch2(batch_size, documents, ids):
            ret_ids += self._add_documents(docs_, ids_)
        return ret_ids

    def _add_documents(self, documents: List[Document], ids: List[str] | None = None) -> List[str]:
        ids = ids or [doc.id for doc in documents]
        metadatas = [{**doc.metadata, "others": doc.model_dump_json(exclude={"id", "text", "metadata"})}
                     for doc in documents]
        texts_to_embed = [doc.get_content(MetadataMode.EMBED) for doc in documents]
        embeddings = self.embedding_function.embed_documents(texts_to_embed)
        self.collection.upsert(metadatas=metadatas,  # type: ignore[arg-type]
                               embeddings=embeddings,  # type: ignore[arg-type]
                               documents=[doc.text for doc in documents],
                               ids=ids)
        return ids

    def search_with_score(self,
                          query: str | None = None,
                          embedding: List[float] | None = None,
                          top_k: int = 4,
                          score_threshold: float | None = None,
                          where: Where | None = None,
                          where_document: WhereDocument | None = None,
                          **kwargs: Any
                          ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        if not embedding:
            assert query
            embedding = self.embedding_function.embed_query(query)
        results = self.collection.query(query_embeddings=[embedding],  # type: ignore[arg-type]
                                        n_results=top_k,
                                        where=where,
                                        where_document=where_document,
                                        **kwargs)

        doc_and_scores = []
        for doc_id, text, metadata, distance in zip(results["ids"][0],
                                                    results["documents"][0],  # type: ignore[index]
                                                    results["metadatas"][0],  # type: ignore[index]
                                                    results["distances"][0]):  # type: ignore[index]
            metadata = dict(metadata)
            others = json.loads(metadata.pop("others"))
            score = 1 - distance
            if score_threshold is None or score >= score_threshold:
                doc_and_scores.append((Document(id=doc_id, text=text, metadata=metadata, **others), score))
        return doc_and_scores

    def mmr_search_with_score(self,
                              query: str | None = None,
                              embedding: List[float] | None = None,
                              top_k: int = 5,
                              score_threshold: float | None = None,
                              fetch_k: int = 20,
                              lambda_mult: float = 0.5,
                              where: Where | None = None,
                              where_document: WhereDocument | None = None,
                              **kwargs: Any
                              ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        if not embedding:
            assert query
            embedding = self.embedding_function.embed_query(query)

        results = self.collection.query(query_embeddings=embedding,
                                        n_results=fetch_k,
                                        where=where,
                                        where_document=where_document,
                                        include=["metadatas", "documents", "distances", "embeddings"],
                                        **kwargs)

        mmr_selected = mmr_top_k(np.array(embedding, dtype=np.float32),
                                 results["embeddings"][0],  # type: ignore[index, arg-type]
                                 similarity_fn=self.similarity_fn,
                                 top_k=top_k,
                                 lambda_mult=lambda_mult)
        doc_and_scores = []
        for index, score in zip(*mmr_selected):
            doc_id = results["ids"][0][index]
            text = results["documents"][0][index]  # type: ignore[index]
            metadata = dict(results["metadatas"][0][index])  # type: ignore[index]
            others = json.loads(metadata.pop("others"))
            if score_threshold is None or score >= score_threshold:
                doc_and_scores.append((Document(id=doc_id, text=text, metadata=metadata, **others), score))
        return doc_and_scores

    def delete(self,
               ids: List[str] | None = None,
               where: Where | None = None,
               where_document: WhereDocument | None = None,
               **kwargs: Any) -> None:
        self.collection.delete(ids=ids, where=where, where_document=where_document)

    def delete_all(self):
        name = self.collection.name
        metadata = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name,
                                                               metadata=metadata,
                                                               embedding_function=None)

    def get(self,
            ids: List[str] | None = None,
            limit: int | None = None,
            offset: int | None = None,
            where: Where | None = None,
            where_document: WhereDocument | None = None,
            **kwargs: Any) -> List[Document]:
        results = self.collection.get(ids=ids,
                                      limit=limit,
                                      offset=offset,
                                      where=where,
                                      where_document=where_document,
                                      **kwargs)
        docs = []
        for doc_id, text, metadata in zip(results["ids"],
                                          results["documents"] or repeat(None),  # type: ignore[index]
                                          results["metadatas"] or repeat(None)):  # type: ignore[index]
            metadata = dict(metadata)
            others = json.loads(metadata.pop("others"))
            docs.append(Document(id=doc_id, text=text, metadata=metadata, **others))
        return docs

    class Config:
        arbitrary_types_allowed = True
