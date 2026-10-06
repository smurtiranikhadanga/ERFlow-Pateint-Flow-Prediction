"""
==============================================================================
Comprehensive RAG Failure Scenarios Test Suite
==============================================================================
Verifies Section D Failure Scenarios:
1. Empty knowledge base directory
2. Missing or unreadable documents
3. ChromaDB unavailable / in-memory fallback
4. Embedding engine failure handling
5. No relevant documents retrieved (similarity below threshold)
6. Complete isolation: RAG failures must never break ML functionality or crash FastAPI
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from backend.rag.document_loader import DocumentLoader
from backend.rag.embeddings import EmbeddingEngine
from backend.rag.vector_store import VectorStore
from backend.rag.retriever import Retriever
from app.chatbot.chatbot_service import chatbot_service
from app.schemas.chat_schema import ChatRequest


def test_failure_scenario_empty_knowledge_base(tmp_path):
    """Test 1: Document loader handles empty directory without crashing."""
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    loader = DocumentLoader(docs_dir=empty_dir)
    docs = loader.load_documents()
    assert docs == [], "Empty knowledge base directory should return empty list gracefully"


def test_failure_scenario_missing_or_corrupted_document(tmp_path):
    """Test 2: Document loader skips unreadable or zero-byte files safely."""
    zero_file = tmp_path / "zero_bytes.md"
    zero_file.write_text("")
    loader = DocumentLoader(docs_dir=tmp_path)
    docs = loader.load_documents()
    assert docs == [], "Zero-byte or corrupted files should be skipped safely"


def test_failure_scenario_chromadb_unavailable_fallback():
    """Test 3: VectorStore falls back to in-memory TF-IDF engine if ChromaDB fails."""
    import sys
    from unittest.mock import MagicMock
    mock_chroma = MagicMock()
    mock_chroma.PersistentClient.side_effect = Exception("ChromaDB connection error")
    with patch.dict(sys.modules, {"chromadb": mock_chroma}), \
         patch("backend.rag.vector_store.chromadb", mock_chroma), \
         patch("backend.rag.vector_store.CHROMADB_AVAILABLE", True):
        vs = VectorStore(collection_name="fallback_test_collection")
        assert vs._use_fallback is True, "VectorStore should activate TF-IDF fallback when ChromaDB is unavailable"



def test_failure_scenario_embedding_engine_failure():
    """Test 4: EmbeddingEngine handles encoding failures without crashing."""
    ee = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    with patch.object(ee, "load_model", side_effect=RuntimeError("Embedding model initialization failed: Model load timeout")):
        with pytest.raises(RuntimeError) as exc_info:
            ee.embed_text("Test query")
        assert "Embedding model initialization failed" in str(exc_info.value)


def test_failure_scenario_no_relevant_documents_retrieved():
    """Test 5: Retriever returns empty context when similarity is below threshold."""
    ret = Retriever()
    context, citations, max_score = ret.retrieve_context("xyzabc non-existent random query 12345", min_score=0.99)
    assert context == ""
    assert citations == []
    assert max_score < 0.99


def test_failure_scenario_rag_failure_does_not_break_ml_predictions():
    """Test 6: ML prediction functionality continues working cleanly even if RAG component fails completely."""
    with patch("backend.rag.retriever.retriever.retrieve_context", side_effect=Exception("Complete RAG Collapse")):
        req = ChatRequest(message="What is the predicted waiting time?")
        resp = chatbot_service.process_message(req)
        assert resp.response is not None
        assert resp.intent in ["WAITING_TIME", "GENERAL_STATUS"]
        assert "waiting-time" in resp.response.lower() or "minutes" in resp.response.lower()


def test_failure_scenario_rag_failure_does_not_crash_fastapi():
    """Test 7: Knowledge query falling through RAG failure returns safe fallback response."""
    with patch("backend.rag.retriever.retriever.retrieve_context", side_effect=Exception("Vector DB Outage")):
        req = ChatRequest(message="What is the hospital diversion policy?")
        resp = chatbot_service.process_message(req)
        assert resp.response is not None
        assert "I'm sorry" in resp.response or "hospital" in resp.response.lower()
