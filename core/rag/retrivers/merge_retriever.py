from collections import defaultdict
from typing import List, Dict, Tuple

from core.rag.document.document import Document
from core.rag.retrivers.retriever import Retriever
from core.rag.vectorstore.vectorstore import VectorStore


class MergeRetriever(Retriever):
    vectorstore: VectorStore
    base_retriever: Retriever
    merge_threshold_ratio: float = 0.5
    """Do merge if the current children ratio >= merge_threshold_ratio."""

    def invoke_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        doc_and_scores = self.base_retriever.invoke_with_scores(query)
        docs = [item[0] for item in doc_and_scores]
        score_dict = {doc.id: score for doc, score in doc_and_scores}
        changed = True
        while changed:
            docs, score_dict, changed1 = self._fill_middle_docs(docs, score_dict)
            docs, score_dict, changed2 = self._merge_to_parent(docs, score_dict)
            changed = changed1 or changed2

        docs.sort(key=lambda x: score_dict[x.id], reverse=True)
        return [(doc, score_dict[doc.id]) for doc in docs]

    def _merge_to_parent(self, docs: List[Document], score_dict: Dict[str, float]
                         ) -> Tuple[List[Document], Dict[str, float], bool]:  # [, score dict, is change]
        parent_docs_dict: Dict[str, Document] = {}
        parent_child_dict: Dict[str, List[str]] = defaultdict(list)

        # Collect doc's parents
        for doc in docs:
            if doc.parent is None:
                continue

            parent_doc_id = doc.parent.doc_id
            # get parent doc
            if parent_doc_id not in parent_docs_dict:
                parent_docs_dict[parent_doc_id] = self.vectorstore.get([parent_doc_id])[0]
            # add child doc
            parent_child_dict[parent_doc_id].append(doc.id)

        cur_doc_ids = set([doc.id for doc in docs])

        # Merge children to parent
        doc_ids_to_delete = []
        doc_to_add: List[Document] = []
        for parent_doc_id, child_doc_ids in parent_child_dict.items():
            # Deletion has higher priority.
            if parent_doc_id in doc_ids_to_delete:
                continue

            # Only delete children if it's parent in current retriever response.
            if parent_doc_id in cur_doc_ids:
                doc_ids_to_delete.extend(child_doc_ids)
                continue

            parent_doc = parent_docs_dict[parent_doc_id]
            ratio = len(child_doc_ids) / len(parent_doc.children)
            if ratio < self.merge_threshold_ratio:
                continue

            doc_ids_to_delete.extend(child_doc_ids)
            doc_to_add.append(parent_doc)

            # update score
            avg_score = sum(
                [score_dict.get(doc_id, 0.0) for doc_id in child_doc_ids]
            ) / len(child_doc_ids)
            score_dict[parent_doc.id] = avg_score

        # add docs
        docs.extend(doc_to_add)

        # delete docs
        docs = [doc for doc in docs if doc.id not in doc_ids_to_delete]
        for doc_id in doc_ids_to_delete:
            score_dict.pop(doc_id)
        return docs, score_dict, len(doc_ids_to_delete) > 0 or len(doc_to_add) > 0

    def _fill_middle_docs(self, docs: List[Document], score_dict: Dict[str, float]
                          ) -> Tuple[List[Document], Dict[str, float], bool]:  # [, score dict, is change]
        """If a doc's left and right docs are selected, then add the middle doc."""
        doc_dict: Dict[str, Document] = {}
        prev_ids = set()
        for doc in docs:
            doc_dict[doc.id] = doc
            if doc.prev:
                prev_ids.add(doc.prev.doc_id)

        middle_docs: List[Document] = []
        for doc in docs:
            if doc.next and doc.next.doc_id not in doc_dict and doc.next.doc_id in prev_ids:
                middle_doc = self.vectorstore.get([doc.next.doc_id])[0]
                middle_docs.append(middle_doc)
                assert middle_doc.next
                right_doc_id = middle_doc.next.doc_id
                avg_score = (score_dict[doc.id] + score_dict[right_doc_id]) / 2
                score_dict[middle_doc.id] = avg_score

        docs.extend(middle_docs)
        return docs, score_dict, len(middle_docs) > 0

    class Config:
        arbitrary_types_allowed = True
