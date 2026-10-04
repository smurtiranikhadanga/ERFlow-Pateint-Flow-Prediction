"""
==============================================================================
Knowledge Retriever Module
==============================================================================
Executes semantic retrieval against the ChromaDB vector store for natural
language queries. 

Returns:
- Retrieved chunk text content
- Full source metadata (filename, doc title, section header)
- Relevance similarity scores
- Formatted cited context passages

Configurable top_k parameter per query.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from .config import rag_settings
from .vector_store import vector_store, VectorStore

logger = logging.getLogger("erflow.rag.retriever")


class Retriever:
    """
    Semantic retrieval engine querying the persistent ChromaDB vector store
    for domain knowledge passages.
    """

    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or vector_store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Backwards-compatible alias for retrieve_context.
        """
        return self.retrieve_context(query=query, top_k=top_k, min_score=min_score)

    def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant document chunks matching a natural language query.

        Args:
            query (str): Natural language query (e.g., "What are ESI level 1 triage guidelines?").
            top_k (Optional[int]): Number of top relevant chunks to retrieve.
            min_score (Optional[float]): Minimum similarity confidence score threshold.

        Returns:
            List[Dict[str, Any]]: List of dictionary objects containing:
                - 'text': Chunk text passage
                - 'metadata': Source document metadata dictionary
                - 'score': Similarity relevance score (0.0 to 1.0)
                - 'source': Source filename
        """
        if not query or not query.strip():
            logger.warning("[Retriever] Empty query string provided.")
            return []

        top_k = top_k or rag_settings.TOP_K_RESULTS
        min_score = min_score if min_score is not None else rag_settings.MIN_SIMILARITY_SCORE

        if hasattr(self.store, "ensure_indexed"):
            try:
                self.store.ensure_indexed()
            except Exception as e:
                logger.warning(f"[Retriever] Vector store indexing note: {e}")

        results = self.store.similarity_search(query=query.strip(), top_k=top_k, min_score=min_score)
        return results

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Retrieves matching knowledge base passages and returns a clean formatted context string,
        a list of citations, and the maximum confidence relevance score.

        Returns:
            Tuple[str, List[Dict[str, Any]], float]:
                - formatted_context_string
                - citations_list
                - max_relevance_score
        """
        results = self.retrieve_chunks(query=query, top_k=top_k, min_score=min_score)

        if not results:
            return "", [], 0.0

        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        max_score = 0.0

        for res in results:
            text = res.get("text", "")
            score = res.get("score", 0.0)
            meta = res.get("metadata", {})
            source = meta.get("source", "Knowledge Base")
            section = meta.get("section_title", "General")

            if score > max_score:
                max_score = score

            context_blocks.append(text)
            citations.append({
                "source": source,
                "section": section,
                "doc_title": meta.get("doc_title", source),
                "score": score,
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)
        return formatted_context, citations, max_score


# Global singleton instance
retriever = Retriever()
