from typing import List, Callable, Sequence

from auto_flow.core.rag.document.document import Document
from auto_flow.core.rag.splitter.text_splitter import TextSplitter, DocumentSplitter
from auto_flow.core.utils.utils import filter_kwargs_by_method


class HierarchicalTextSplitter(DocumentSplitter):
    doc_splitters: List[DocumentSplitter]
    """Node splitters in order of level."""

    chunk_sizes: List[int] | None = None
    """"The chunk sizes to use when splitting documents, in order of level."""

    def __init__(self,
                 doc_splitters: List[DocumentSplitter] | None = None,
                 chunk_sizes: Sequence[int] = (2048, 512, 128),
                 chunk_overlap: int | None = None,
                 sentence_seps: str | None = None,
                 secondary_seps: str | None = None,
                 token_seps: str | None = None,
                 separate_fn_list: List[Callable] | None = None,
                 token_length_fn: Callable[[str], int] | None = None):
        if doc_splitters is None:
            kwargs = filter_kwargs_by_method(TextSplitter.__init__, locals(), exclude_none=True)
            doc_splitters = [TextSplitter(chunk_size=chunk_size, **kwargs) for chunk_size in chunk_sizes]

        super().__init__(doc_splitters=doc_splitters, chunk_sizes=chunk_sizes)  # type: ignore[call-arg]

    def split_text(self, text: str, chunk_size: int | None = None) -> List[str]:
        raise NotImplementedError

    def split_document(self, documents: List[Document]) -> List[Document]:
        ret_documents = []
        for level in range(len(self.doc_splitters)):
            documents = self.split_in_level(documents, level)
            ret_documents.extend(documents)
        return ret_documents

    def split_in_level(self, documents: List[Document], level: int) -> List[Document]:
        sub_docs = []
        prev_doc: Document | None = None
        for doc in documents:
            cur_sub_docs = self.doc_splitters[level].split_document([doc])

            # todo 应该没有必要再把不同文档首尾连接了
            # if prev_doc:
            #     prev_doc.add_next(cur_sub_docs[0])
            # prev_doc = cur_sub_docs[-1]

            sub_docs.extend(cur_sub_docs)

            if level > 0:
                for sub_doc in cur_sub_docs:
                    doc.add_child(sub_doc)
        return sub_docs
