"""
==============================================================================
Document Loader Subsystem
==============================================================================
Handles multi-format document ingestion (Markdown, Plain Text, PDF), text
extraction, metadata preservation, and error handling for the ERFlow RAG pipeline.

Isolated from FastAPI server startup and ML prediction routines.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .config import rag_settings

logger = logging.getLogger("erflow.rag.document_loader")


class Document:
    """
    Structured representation of an ingested knowledge base document with metadata.
    """
    def __init__(
        self,
        content: str,
        source: str,
        doc_title: str,
        file_path: Path,
        file_type: str,
        file_size_bytes: int,
        page_count: int = 1,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.source = source
        self.doc_title = doc_title
        self.file_path = file_path
        self.file_type = file_type
        self.file_size_bytes = file_size_bytes
        self.page_count = page_count
        self.metadata = {
            "source": source,
            "doc_title": doc_title,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "page_count": page_count,
            "file_path": str(file_path),
            **(additional_metadata or {}),
        }

    def __repr__(self) -> str:
        return (
            f"<Document title='{self.doc_title}' type='{self.file_type}' "
            f"pages={self.page_count} bytes={self.file_size_bytes}>"
        )


class DocumentLoader:
    """
    Robust document loader supporting Markdown (.md), Plain Text (.txt), and PDF (.pdf).
    Includes encoding fallback, metadata generation, and non-blocking error handling.
    """

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

    def __init__(self, docs_dir: Optional[Union[Path, str]] = None):
        self.docs_dir = Path(docs_dir) if docs_dir else rag_settings.KNOWLEDGE_BASE_DIR
        self.auto_ingest_on_startup = False  # Explicit safety flag: No startup auto-ingest

    def load_documents(self) -> List[Document]:
        """
        Scans the knowledge base directory and ingests all supported documents.

        Returns:
            List[Document]: List of successfully loaded Document objects.
        """
        documents: List[Document] = []

        if not self.docs_dir.exists():
            logger.warning(
                f"[DocumentLoader] Knowledge base directory does not exist: {self.docs_dir}. "
                f"Creating directory..."
            )
            try:
                self.docs_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"[DocumentLoader] Failed to create knowledge base directory: {e}")
                return documents

        # Scan for supported files
        all_files = [
            p for p in self.docs_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        if not all_files:
            logger.info(f"[DocumentLoader] No knowledge documents found in {self.docs_dir}")
            return documents

        logger.info(f"[DocumentLoader] Found {len(all_files)} potential documents for ingestion.")

        for file_path in all_files:
            doc = self.load_single_file(file_path)
            if doc:
                documents.append(doc)

        logger.info(f"[DocumentLoader] Successfully ingested {len(documents)} / {len(all_files)} documents.")
        return documents

    def load_single_file(self, file_path: Union[Path, str]) -> Optional[Document]:
        """
        Ingests a single file with format-specific text extraction and error recovery.
        """
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"[DocumentLoader] Target file does not exist or is invalid: {file_path}")
            return None

        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"[DocumentLoader] Unsupported file extension '{ext}' for file {file_path.name}")
            return None

        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.warning(f"[DocumentLoader] Skipping empty 0-byte file: {file_path.name}")
            return None

        doc_title = file_path.stem.replace("_", " ").replace("-", " ").title()

        try:
            if ext == ".md" or ext == ".txt":
                return self._extract_text_file(file_path, ext, file_size, doc_title)
            elif ext == ".pdf":
                return self._extract_pdf_file(file_path, file_size, doc_title)
        except Exception as e:
            logger.error(f"[DocumentLoader] Unexpected failure ingesting {file_path.name}: {e}", exc_info=True)
            return None

        return None

    def _extract_text_file(self, file_path: Path, ext: str, file_size: int, doc_title: str) -> Optional[Document]:
        """Reads Markdown or plain text files using multi-encoding fallback."""
        content = None
        encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc, errors="strict") as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"[DocumentLoader] Encoding read error for {file_path.name} ({enc}): {e}")

        if content is None:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"[DocumentLoader] Fallback read error for {file_path.name}: {e}")

        if not content or not content.strip():
            logger.warning(f"[DocumentLoader] File yielded no readable text content: {file_path.name}")
            return None

        # Clean null bytes if any
        clean_content = content.replace("\x00", "").strip()

        return Document(
            content=clean_content,
            source=file_path.name,
            doc_title=doc_title,
            file_path=file_path,
            file_type=ext,
            file_size_bytes=file_size,
            page_count=1,
        )

    def _extract_pdf_file(self, file_path: Path, file_size: int, doc_title: str) -> Optional[Document]:
        """Extracts text content from PDF documents page by page."""
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(str(file_path))
            page_texts = []
            page_count = len(reader.pages)

            for i, page in enumerate(reader.pages, start=1):
                txt = page.extract_text()
                if txt and txt.strip():
                    page_texts.append(f"[Page {i}]\n{txt.strip()}")

            if not page_texts:
                logger.warning(f"[DocumentLoader] PDF file yielded no text content: {file_path.name}")
                return None

            combined_content = "\n\n".join(page_texts)

            return Document(
                content=combined_content,
                source=file_path.name,
                doc_title=doc_title,
                file_path=file_path,
                file_type=".pdf",
                file_size_bytes=file_size,
                page_count=page_count,
            )

        except ImportError:
            logger.warning(
                f"[DocumentLoader] PDF file encountered '{file_path.name}', but optional dependency 'pypdf' "
                f"is not installed. Install pypdf to enable PDF document extraction."
            )
            return None
        except Exception as e:
            logger.error(f"[DocumentLoader] Failed to extract PDF {file_path.name}: {e}")
            return None


# Global singleton instance
document_loader = DocumentLoader()
