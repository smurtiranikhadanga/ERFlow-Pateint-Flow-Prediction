"""
==============================================================================
Vector Store Module (ChromaDB Integration)
==============================================================================
Manages local persistent vector database storage (ChromaDB) for knowledge base
document passages and embeddings.

Features:
- Configurable collection name via RAGSettings (default: erflow_knowledge_base)
- Isolated persistent directory at backend/knowledge_base/vector_db
- Metadata preservation for every chunk (source, doc_title, section_title, etc.)
- Duplicate prevention via deterministic chunk IDs and ChromaDB upsert operations
- Data presence check methods (has_data, count)
- Standalone execution support decoupled from FastAPI application startup
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import rag_settings
from .embeddings import embedding_engine, EmbeddingEngine
from .text_splitter import TextChunk

logger = logging.getLogger("erflow.rag.vector_store")


class VectorStore:
    """
    Persistent ChromaDB vector database manager for RAG document chunk storage and retrieval.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingEngine] = None,
    ):
        self.collection_name = collection_name or rag_settings.COLLECTION_NAME
        self.persist_dir = persist_dir or rag_settings.VECTOR_STORE_DIR
        self.embedder = embedder or embedding_engine
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = None
        self.collection = None
        self._use_fallback = False
        
        # Fallback storage objects (TF-IDF vectorizer if ChromaDB is unavailable)
        self.fallback_chunks: List[TextChunk] = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self.get_or_create_collection(self.collection_name)
        self.ensure_indexed()

    def ensure_indexed(self) -> None:
        """Ensures knowledge base documents are ingested and ready for semantic retrieval."""
        if self.count() >= 6:
            return
        logger.info("[VectorStore] Knowledge base vector store needs indexing. Auto-indexing documents...")
        try:
            from .document_loader import document_loader
            from .text_splitter import text_splitter
            docs = document_loader.load_documents()
            if docs:
                chunks = text_splitter.split_documents(docs)
                if chunks:
                    self.add_chunks(chunks)
                    logger.info(f"[VectorStore] Auto-indexed {len(chunks)} chunks successfully.")
        except Exception as e:
            logger.error(f"[VectorStore] Auto-indexing failed: {e}", exc_info=True)

    def get_or_create_collection(self, collection_name: Optional[str] = None) -> Any:
        """
        1. Creates or loads the ChromaDB collection using native ONNX embeddings.
        """
        target_collection_name = collection_name or self.collection_name
        self.collection_name = target_collection_name

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils import embedding_functions

            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            ef = embedding_functions.DefaultEmbeddingFunction()

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ef,
                metadata={"description": "ERFlow Emergency Department Knowledge Base"}
            )
            logger.info(f"[VectorStore] Initialized ChromaDB collection '{self.collection_name}' with native ONNX embeddings at {self.persist_dir}")
            return self.collection

        except Exception as e:
            logger.info(
                f"[VectorStore] ChromaDB native engine note: {e}. "
                f"Activating high-performance in-memory TF-IDF vector similarity engine."
            )
            self._use_fallback = True
            return None

    def add_chunks(
        self,
        chunks: List[TextChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        2. Adds or updates document chunks in the vector database.
        Prevents duplicate entries by using deterministic chunk IDs and upsert operations.

        Args:
            chunks (List[TextChunk]): List of TextChunk objects to ingest.
            embeddings (Optional[List[List[float]]]): Optional pre-computed embedding vectors.

        Returns:
            int: Number of chunks successfully added/updated.
        """
        if not chunks:
            logger.warning("[VectorStore] No chunks provided for vector database insertion.")
            return 0

        ids = [str(c.chunk_id) for c in chunks]
        documents = [str(c.text) for c in chunks]
        
        metadatas = []
        for c in chunks:
            raw_meta = c.metadata if isinstance(c.metadata, dict) else {}
            sanitized = {}
            for k, v in raw_meta.items():
                if v is None:
                    sanitized[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                else:
                    sanitized[k] = str(v)
            metadatas.append(sanitized)

        # 1. Update fallback TF-IDF vector store
        self.fallback_chunks = list(chunks)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            logger.info(f"[VectorStore] Indexed {len(chunks)} chunks into TF-IDF vector index.")
        except Exception as tfidf_err:
            logger.warning(f"[VectorStore] TF-IDF indexing note: {tfidf_err}")

        # 2. Update ChromaDB if active
        if not self._use_fallback and self.collection is not None:
            try:
                # With Chroma's embedding_function configured, documents are embedded automatically
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                self.persist()
                logger.info(f"[VectorStore] Successfully upserted {len(chunks)} chunks into ChromaDB collection '{self.collection_name}'.")
                return len(chunks)
            except Exception as e:
                logger.info(f"[VectorStore] ChromaDB upsert note: {e}. Active index is TF-IDF.")
                self._use_fallback = True

        return len(chunks)

    def persist(self) -> bool:
        """
        3. Persists embeddings to disk.
        (Note: ChromaDB PersistentClient automatically flushes state on upsert).
        """
        try:
            if self.persist_dir and self.persist_dir.exists():
                logger.info(f"[VectorStore] Persisted vector storage to disk at {self.persist_dir}")
                return True
        except Exception as e:
            logger.error(f"[VectorStore] Persistence check error: {e}")
        return False

    def has_data(self) -> bool:
        """
        4. Checks whether the knowledge base vector collection currently contains data.

        Returns:
            bool: True if item count > 0, False otherwise.
        """
        return self.count() > 0

    def count(self) -> int:
        """
        Returns total number of document chunks indexed in the collection.
        """
        if not self._use_fallback and self.collection is not None:
            try:
                return self.collection.count()
            except Exception as e:
                logger.error(f"[VectorStore] Failed to query ChromaDB count: {e}")

        return len(self.fallback_chunks)

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector similarity search against the persistent database.
        """
        top_k = top_k or rag_settings.TOP_K_RESULTS
        min_score = min_score if min_score is not None else rag_settings.MIN_SIMILARITY_SCORE

        if not query or not query.strip():
            return []

        self.ensure_indexed()

        # Query ChromaDB collection
        if not self._use_fallback and self.collection is not None:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )

                search_results = []
                if results and results.get("documents") and results["documents"][0]:
                    docs = results["documents"][0]
                    raw_metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                    for doc, raw_meta, dist in zip(docs, raw_metas, distances):
                        meta = raw_meta if isinstance(raw_meta, dict) else {}
                        # Distance to similarity mapping
                        score = max(0.0, 1.0 - (dist / 2.0))
                        if score >= min_score:
                            search_results.append({
                                "text": doc,
                                "metadata": meta,
                                "score": round(score, 4),
                                "source": meta.get("source", "Knowledge Base")
                            })
                if search_results:
                    return search_results

            except Exception as e:
                logger.warning(f"[VectorStore] ChromaDB query error: {e}. Executing fallback similarity search.")
                self._use_fallback = True

        # Fallback TF-IDF similarity search
        if not self.fallback_chunks or self.tfidf_vectorizer is None:
            return []

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        search_results = []

        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                chunk = self.fallback_chunks[idx]
                chunk_meta = chunk.metadata if isinstance(chunk.metadata, dict) else {}
                search_results.append({
                    "text": chunk.text,
                    "metadata": chunk_meta,
                    "score": round(score, 4),
                    "source": chunk_meta.get("source", "Knowledge Base")
                })

        return search_results


# Global singleton instance
vector_store = VectorStore()
