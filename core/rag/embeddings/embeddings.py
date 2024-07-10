from abc import abstractmethod, ABC
from typing import List, Union

import numpy as np
from numpy._typing import NDArray


class Embeddings(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        ...

    @abstractmethod
    def embed_documents(self, text_list: List[str]) -> List[List[float]]:
        ...

    def __call__(self, input: List[str] | NDArray[Union[np.uint, np.int_, np.float_]]) -> List[List[float]]:
        return self.embed_documents(input)
