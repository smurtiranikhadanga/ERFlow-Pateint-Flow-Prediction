"""
==============================================================================
Embedding Service Module
==============================================================================
Provides reusable vector embedding generation using SentenceTransformers
(all-MiniLM-L6-v2) for semantic text search and similarity comparisons.

Features:
- Configurable model name via RAGSettings (default: all-MiniLM-L6-v2)
- Lazy model loading (initializes model strictly on-demand, loaded once per process)
- Vector dimension verification (384-dimensional dense vectors)
- Batch document embedding and query embedding methods
- Cosine similarity computation between text passages
- Completely isolated from TensorFlow, XGBoost, and FastAPI startup routines
"""

import logging
from typing import List, Union, Optional
import numpy as np

from .config import rag_settings

logger = logging.getLogger("erflow.rag.embeddings")


class EmbeddingEngine:
    """
    Reusable embedding service wrapping SentenceTransformers for semantic search.
    Implements lazy initialization so the model is loaded only when first invoked.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or rag_settings.EMBEDDING_MODEL_NAME
        self.target_dimension = rag_settings.EMBEDDING_DIMENSION
        self._model = None
        self._is_loaded = False

    def load_model(self):
        """
        Lazily loads the SentenceTransformer model into memory if not already initialized.
        Falls back to normalized feature hashing if SentenceTransformers is unavailable.
        """
        if self._is_loaded:
            return self._model

        logger.info(f"[EmbeddingEngine] Lazily loading SentenceTransformer model '{self.model_name}'...")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._is_loaded = True
            logger.info(
                f"[EmbeddingEngine] Successfully initialized '{self.model_name}'. "
                f"Vector dimension: {self.dimension}"
            )
        except Exception as e:
            logger.warning(f"[EmbeddingEngine] SentenceTransformer unavailable ({e}). Operating in lightweight feature-hashing vector mode.")
            self._model = None
            self._is_loaded = True

        return self._model

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying model is initialized in memory."""
        return self._is_loaded

    @property
    def dimension(self) -> int:
        """
        Returns the vector embedding dimension (e.g. 384 for all-MiniLM-L6-v2).
        """
        if self._model is not None:
            try:
                if hasattr(self._model, "get_embedding_dimension"):
                    return self._model.get_embedding_dimension()
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    return self._model.get_sentence_embedding_dimension()
            except Exception:
                pass
        return self.target_dimension

    def _fallback_embed(self, text: str) -> List[float]:
        """Generates a normalized 384-dimensional feature hash vector for text."""
        words = [w.strip().lower() for w in text.split() if w.strip()]
        if not words:
            return [0.0] * self.dimension
        vec = np.zeros(self.dimension, dtype=np.float32)
        for w in words:
            idx = abs(hash(w)) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a normalized embedding vector for a single text string.

        Args:
            text (str): Input text or search query.

        Returns:
            List[float]: 384-dimensional vector representation.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        model = self.load_model()
        if model is None:
            return self._fallback_embed(text)

        vector = model.encode(text.strip(), convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates normalized embedding vectors for a list of document passages in batches.

        Args:
            texts (List[str]): List of text passages.
            batch_size (int): Batch size for GPU/CPU vector encoding.

        Returns:
            List[List[float]]: List of 384-dimensional vector embeddings.
        """
        if not texts:
            return []

        clean_texts = [t.strip() if t and t.strip() else " " for t in texts]
        model = self.load_model()
        if model is None:
            return [self._fallback_embed(t) for t in clean_texts]
        
        vectors = model.encode(
            clean_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vectors.tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes cosine similarity between two text strings (range: 0.0 to 1.0).

        Args:
            text1 (str): First text passage.
            text2 (str): Second text passage.

        Returns:
            float: Cosine similarity score (higher indicates greater semantic similarity).
        """
        v1 = np.array(self.embed_text(text1), dtype=np.float32)
        v2 = np.array(self.embed_text(text2), dtype=np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = float(np.dot(v1, v2) / (norm1 * norm2))
        return round(max(0.0, min(1.0, similarity)), 4)


# Singleton instance for reusable embedding service
embedding_engine = EmbeddingEngine()
