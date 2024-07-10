import sys
from typing import Tuple, List, Callable

import numpy as np
from enum import Enum


class SimilarityMode(Enum):
    EUCLIDEAN = 1
    DOT_PRODUCT = 2
    COSINE = 3
    DEFAULT = COSINE


def calc_similarity(query_embedding: np.ndarray,
                    embedding_list: np.ndarray,
                    mode: SimilarityMode = SimilarityMode.DEFAULT
                    ) -> np.ndarray:  # return list of scores
    """Get embedding similarity for batches of embeddings."""

    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    if embedding_list.ndim == 1:
        embedding_list = np.expand_dims(embedding_list, axis=0)

    if mode == SimilarityMode.EUCLIDEAN:
        return -np.linalg.norm(query_embedding - embedding_list, axis=1)
    elif mode == SimilarityMode.DOT_PRODUCT:
        return np.sum(query_embedding * embedding_list, axis=1)
    else:
        product = np.sum(query_embedding * embedding_list, axis=1)
        norm1 = np.linalg.norm(query_embedding, axis=1)
        norm2 = np.linalg.norm(embedding_list, axis=1)
        return product / (norm1 * norm2)


def similarity_top_k(query_embedding: np.ndarray | List[List[float]],
                     embedding_list: np.ndarray | List[List[float]],
                     similarity_fn: Callable[..., np.ndarray] = calc_similarity,
                     top_k: int | None = 5,
                     score_threshold: float = sys.float_info.min,
                     ) -> Tuple[List[int], List[float]]:
    if len(query_embedding) == 0 or len(embedding_list) == 0:
        return [], []
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    if isinstance(embedding_list, list):
        embedding_list = np.array(embedding_list)
    score_array = similarity_fn(query_embedding, embedding_list)
    top_k = min(top_k or len(score_array), int(np.sum(score_array >= score_threshold)))
    top_k_indices = np.argpartition(score_array, -top_k, axis=None)[-top_k:]
    top_k_indices = top_k_indices[np.argsort(score_array[top_k_indices])][::-1]
    scores = score_array[top_k_indices].tolist()
    return top_k_indices.tolist(), scores


def mmr_top_k(query_embedding: np.ndarray | List[float],
              embedding_list: np.ndarray | List[List[float]],
              similarity_fn: Callable[..., np.ndarray] = calc_similarity,
              lambda_mult: float = 0.5,
              top_k: int | None = 5) -> Tuple[List[int], List[float]]:
    """mmr = maximal marginal relevance"""

    if min(top_k, len(embedding_list)) <= 0:
        return [], []
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    if isinstance(embedding_list, list):
        embedding_list = np.array(embedding_list)
    score_array = similarity_fn(query_embedding, embedding_list)
    most_similar = int(np.argmax(score_array))
    top_k_indices, scores = [most_similar], [score_array[most_similar]]
    selected = np.array([embedding_list[most_similar]])
    while len(top_k_indices) < min(top_k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        for i, query_score in enumerate(score_array):
            if i in top_k_indices:
                continue
            similarity_to_selected = similarity_fn(embedding_list[i], selected)
            redundant_score = max(similarity_to_selected)
            equation_score = (
                    lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        top_k_indices.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
        scores.append(best_score)
    return top_k_indices, scores
