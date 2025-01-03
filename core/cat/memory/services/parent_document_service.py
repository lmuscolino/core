from enum import Enum
from typing import List

from langchain.retrievers import ParentDocumentRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class CustomParentDocumentRetriever(ParentDocumentRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)

        # save subdocs into run manager
        metadata_runmanager = {
            "sub_docs": sub_docs,
        }

        setattr(run_manager, "meta_sub_docs", metadata_runmanager)

        return [d for d in docs if d is not None]
