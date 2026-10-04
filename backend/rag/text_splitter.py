"""
==============================================================================
Text Splitter Module
==============================================================================
Splits full knowledge base documents into meaningful, overlapping text chunks
suitable for vector embedding and semantic retrieval.

Features:
- Section-aware splitting (preserves Markdown section headings)
- Sliding window word chunking with configurable chunk_size and chunk_overlap
- Full metadata preservation (source filename, document title, section title,
  chunk index, character count, word count)
- Suitable for student project demonstration and easy maintenance.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from .config import rag_settings
from .document_loader import Document

logger = logging.getLogger("erflow.rag.text_splitter")


class TextChunk:
    """
    Represents a single text passage chunk with metadata for vector retrieval.
    """
    def __init__(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        chunk_index: int,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata
        self.chunk_index = chunk_index

    def __repr__(self) -> str:
        words = len(self.text.split())
        return f"<TextChunk id='{self.chunk_id}' words={words} source='{self.metadata.get('source')}'>"

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary representation of the chunk."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
        }


class TextSplitter:
    """
    Modular text splitter for breaking down documents into overlapping passages.
    
    Parameters:
        chunk_size (int): Target maximum number of words per chunk (default from config).
        chunk_overlap (int): Number of overlapping words between consecutive chunks (default from config).
    """
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or rag_settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or rag_settings.CHUNK_OVERLAP

    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Splits a list of Document objects into indexed TextChunk passages.

        Args:
            documents (List[Document]): Ingested Document objects.

        Returns:
            List[TextChunk]: Generated text chunks with full metadata.
        """
        all_chunks: List[TextChunk] = []

        if not documents:
            logger.warning("[TextSplitter] No documents provided to split.")
            return all_chunks

        for doc in documents:
            doc_chunks = self.split_single_document(doc)
            all_chunks.extend(doc_chunks)

        logger.info(f"[TextSplitter] Generated {len(all_chunks)} text chunks from {len(documents)} documents.")
        return all_chunks

    def split_single_document(self, doc: Document) -> List[TextChunk]:
        """
        Splits a single document using section-aware headers and sliding window overlap.
        """
        chunks: List[TextChunk] = []
        if not doc or not doc.content or not doc.content.strip():
            return chunks

        # 1. Section-aware splitting by major Markdown headers (##) with automatic merging of short preamble headers
        raw_sections = re.split(r'\n(?=##\s)', doc.content)
        sections = []
        for s in raw_sections:
            s_clean = s.strip()
            if not s_clean:
                continue
            if sections and len(sections[-1].split()) < 40:
                sections[-1] = sections[-1] + "\n\n" + s_clean
            else:
                sections.append(s_clean)

        chunk_counter = 0

        for section in sections:
            section_str = section.strip()
            if not section_str:
                continue

            # Extract section title from first header line if available
            lines = section_str.splitlines()
            section_title = doc.doc_title
            if lines and lines[0].startswith('#'):
                section_title = lines[0].lstrip('#').strip()

            # 2. Sliding window word chunking within section
            sub_passages = self._sliding_window_words(section_str, self.chunk_size, self.chunk_overlap)

            for passage in sub_passages:
                clean_passage = passage.strip()
                if len(clean_passage) < 20:  # Skip tiny fragments
                    continue

                chunk_counter += 1
                chunk_id = f"{doc.file_path.stem}_chunk_{chunk_counter}"
                words = clean_passage.split()

                metadata = {
                    "source": doc.source,
                    "doc_title": doc.doc_title,
                    "section_title": section_title,
                    "file_type": doc.file_type,
                    "file_size_bytes": doc.file_size_bytes,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_counter,
                    "word_count": len(words),
                    "char_count": len(clean_passage),
                }

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=clean_passage,
                        metadata=metadata,
                        chunk_index=chunk_counter,
                    )
                )

        return chunks

    def _sliding_window_words(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Applies a sliding window algorithm over words to enforce chunk_size and chunk_overlap.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        passages = []
        step = max(1, chunk_size - overlap)
        i = 0

        while i < len(words):
            window_words = words[i : i + chunk_size]
            passages.append(" ".join(window_words))
            i += step
            # Exit if remaining words are fully covered by overlap
            if i + overlap >= len(words):
                break

        return passages


# Singleton instance for easy import
text_splitter = TextSplitter()
