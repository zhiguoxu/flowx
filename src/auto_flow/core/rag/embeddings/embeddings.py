from abc import abstractmethod, ABC
from typing import List


class Embeddings(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def embed_documents(self, text_list: List[str]) -> List[List[float]]:
        """
        todo If documents are too long, refer to
        https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        and OpenAIEmbeddings._get_len_safe_embeddings
        """
