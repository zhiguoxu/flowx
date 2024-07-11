import uuid
from typing import Dict, Callable, List, Tuple, Any

import numpy as np
from chromadb import Settings
from chromadb.api.client import Client
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field

from core.rag.document.document import Document
from core.rag.embeddings.embeddings import Embeddings
from core.rag.embeddings.huggingface.hf_embedding import HuggingfaceEmbeddings
from core.rag.utils import batch3
from core.rag.vectorstore.utils import mmr_top_k, calc_similarity
from core.rag.vectorstore.vectorstore import VectorStore
from core.utils.utils import filter_kwargs_by_pydantic, filter_kwargs_by_method

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

    def add_texts(self,
                  texts: List[str],
                  metadatas: List[dict] | None = None,
                  ids: List[str] | None = None,
                  batch_size: int = 100) -> List[str]:
        ret_ids = []
        for texts_, metadatas_, ids_ in batch3(batch_size, texts, metadatas, ids):
            ret_ids += self._add_texts(texts_, metadatas_, ids_)
        return ret_ids

    def _add_texts(self,
                   texts: List[str],
                   metadatas: List[dict] | None = None,
                   ids: List[str] | None = None) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedding_function.embed_documents(texts)
        if not metadatas:
            self.collection.upsert(embeddings=embeddings, documents=texts, ids=ids)
            return ids

        assert len(texts) == len(metadatas)
        self.collection.upsert(metadatas=metadatas,
                               embeddings=embeddings,
                               documents=texts,
                               ids=ids)
        return ids

    def search_with_score(self,
                          query: str | None = None,
                          embedding: List[float] | None = None,
                          k: int = 4,
                          filters: Dict[str, str] | None = None,
                          where_document: Dict[str, str] = None,
                          **kwargs: Any
                          ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        if not embedding:
            assert query
            embedding = self.embedding_function.embed_query(query)
        results = self.collection.query(query_embeddings=[embedding],
                                        n_results=k,
                                        where=filters,
                                        where_document=where_document,
                                        **kwargs)

        return [(Document(text=result[0], metadata=result[1] or {}, id=result[2]), 1 - result[3])
                for result in zip(results["documents"][0],
                                  results["metadatas"][0],
                                  results["ids"][0],
                                  results["distances"][0])]

    def mmr_search_with_score(self,
                              query: str | None = None,
                              embedding: List[float] | None = None,
                              k: int = 5,
                              fetch_k: int = 20,
                              lambda_mult: float = 0.5,
                              filters: Dict[str, str] | None = None,
                              where_document: Dict[str, str] | None = None,
                              **kwargs: Any
                              ) -> List[Tuple[Document, float]]:  # [document, similarity score]
        if not embedding:
            assert query
            embedding = self.embedding_function.embed_query(query)

        results = self.collection.query(query_embeddings=embedding,
                                        n_results=fetch_k,
                                        where=filters,
                                        where_document=where_document,
                                        include=["metadatas", "documents", "distances", "embeddings"],
                                        **kwargs)

        mmr_selected = mmr_top_k(np.array(embedding, dtype=np.float32),
                                 results["embeddings"][0],
                                 similarity_fn=self.similarity_fn,
                                 top_k=k,
                                 lambda_mult=lambda_mult)

        return [(Document(text=results["documents"][0][index],
                          metadata=results["metadatas"][0][index] or {},
                          id=results["ids"][0][index]), score)
                for index, score in zip(*mmr_selected)]

    def delete(self,
               ids: List[str] | None = None,
               filters: Dict[str, str] | None = None,
               where_document: Dict[str, str] | None = None) -> None:
        self.collection.delete(ids=ids, where=filters, where_document=where_document)

    def delete_all(self):
        name = self.collection.name
        metadata = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name,
                                                               metadata=metadata,
                                                               embedding_function=None)

    class Config:
        arbitrary_types_allowed = True
