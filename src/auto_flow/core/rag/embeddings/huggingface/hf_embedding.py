from typing import List, Dict
from typing import Any
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from auto_flow.core.rag.embeddings.embeddings import Embeddings
from auto_flow.core.rag.utils import infer_torch_device, to_list, OneOrMany
from auto_flow.core.utils.utils import filter_kwargs_by_pydantic, filter_kwargs_by_method

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_HUGGINGFACE_LENGTH = 512
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "
DEFAULT_QUERY_BGE_INSTRUCTION_EN = "Represent this question for searching relevant passages: "
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


class HuggingfaceEmbeddings(BaseModel, Embeddings):
    model: str = DEFAULT_MODEL_NAME
    client: SentenceTransformer
    # Encode parameters.
    query_instruction: str | None = None
    document_instruction: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    show_progress: bool = False
    multi_process: bool = False  # Run encode() on multiple GPUs.
    extra_encode_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self,
                 model: str = DEFAULT_MODEL_NAME,
                 # Encode parameters.
                 query_instruction: str | None = None,
                 document_instruction: str | None = None,
                 normalize: bool = True,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 show_progress: bool = False,
                 # SentenceTransformer init parameters.
                 device: str | None = None,
                 trust_remote_code: bool = False,
                 cache_folder: str | None = None,
                 max_length: int | None = None,
                 **model_or_encode_kwargs: Any):
        # Init instruction.
        if model.startswith("BAAI/bge-"):
            if "-zh" in model.lower():
                query_instruction = query_instruction or DEFAULT_QUERY_BGE_INSTRUCTION_ZH
            else:
                query_instruction = query_instruction or DEFAULT_QUERY_BGE_INSTRUCTION_EN
        elif "instructor" in model.lower():
            query_instruction = query_instruction or DEFAULT_QUERY_INSTRUCTION

        if "instructor" in model.lower():
            document_instruction = document_instruction or DEFAULT_QUERY_INSTRUCTION

        device = device or infer_torch_device()

        # Collect model init kwargs.
        model_init_kwargs = filter_kwargs_by_method(SentenceTransformer.__init__,
                                                    {**locals(), **model_or_encode_kwargs},
                                                    exclude_none=True)
        client = SentenceTransformer(model, **model_init_kwargs)
        if max_length:
            client.max_seq_length = max_length

        extra_encode_kwargs = filter_kwargs_by_method(client.encode, model_or_encode_kwargs)
        kwargs = filter_kwargs_by_pydantic(HuggingfaceEmbeddings, locals(), exclude_none=True)
        super().__init__(**kwargs)

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, prompt=self.query_instruction)[0]

    def embed_documents(self, text_list: List[str]) -> List[List[float]]:
        return self._embed(text_list, prompt=self.document_instruction)

    def _embed(self, sentences: OneOrMany[str], prompt: str | None = None) -> List[List[float]]:
        sentences = [s.replace("\n", " ") for s in to_list(sentences)]
        if self.multi_process:
            import sentence_transformers
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(sentences, pool, prompt=prompt, batch_size=self.batch_size)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(sentences,
                                            prompt=prompt,
                                            batch_size=self.batch_size,
                                            normalize_embeddings=self.normalize,
                                            show_progress_bar=self.show_progress,
                                            **self.extra_encode_kwargs)  # type: ignore[assignment]
        return embeddings.tolist()

    @property
    def max_length(self):
        return self.client.max_seq_length

    @max_length.setter
    def max_length(self, value: int):
        self.client.max_seq_length = value

    class Config:
        arbitrary_types_allowed = True
