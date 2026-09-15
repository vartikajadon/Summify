import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from pipeline.structure import StructuredDocument, structured_doc_from_dict, export_markdown, export_docx, export_pdf
from pipeline.index import sanitize_collection_name, HAS_CHROMADB, _create_chroma_client, InMemoryVectorStore

logger = logging.getLogger(__name__)

def load_data_root(config_path: str = "config.yaml") -> str:
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("storage", {}).get("data_root", "./data")
        except Exception:
            pass
    return "./data"

def get_session_paths(session_id: str, data_root: Optional[str] = None) -> Dict[str, Path]:
    root_str = data_root or load_data_root()
    base_dir = Path(root_str) / "sessions" / session_id
    uploads_dir = base_dir / "uploads"
    exports_dir = base_dir / "exports"
    return {
        "root_dir": base_dir,
        "uploads_dir": uploads_dir,
        "exports_dir": exports_dir
    }

def create_session(data_root: Optional[str] = None) -> str:
    """
    Generates a unique UUID session_id and initializes per-session directory layout.
    """
    session_id = str(uuid.uuid4())
    paths = get_session_paths(session_id, data_root=data_root)
    
    paths["uploads_dir"].mkdir(parents=True, exist_ok=True)
    paths["exports_dir"].mkdir(parents=True, exist_ok=True)

    logger.info(f"Created session '{session_id}' under directory {paths['root_dir']}")
    return session_id

def export_all(
    session_id: str,
    structured_doc: Optional[Any] = None,
    data_root: Optional[str] = None
) -> Dict[str, bytes]:
    """
    Exports the structured document into Markdown, DOCX, and PDF byte streams,
    saving them to the session exports directory and returning a dict of bytes.
    """
    doc_obj = structured_doc_from_dict(structured_doc)

    md_str = export_markdown(doc_obj)
    md_bytes = md_str.encode("utf-8")
    docx_bytes = export_docx(doc_obj)
    pdf_bytes = export_pdf(doc_obj)

    paths = get_session_paths(session_id, data_root=data_root)
    paths["exports_dir"].mkdir(parents=True, exist_ok=True)

    # Save to disk
    with open(paths["exports_dir"] / "documentation.md", "wb") as f:
        f.write(md_bytes)
    with open(paths["exports_dir"] / "documentation.docx", "wb") as f:
        f.write(docx_bytes)
    with open(paths["exports_dir"] / "documentation.pdf", "wb") as f:
        f.write(pdf_bytes)

    return {
        "markdown": md_bytes,
        "docx": docx_bytes,
        "pdf": pdf_bytes
    }

def delete_session(session_id: str, data_root: Optional[str] = None, storage_dir: Optional[str] = None) -> None:
    """
    Deletes all uploaded files, generated document exports, and the Chroma vector collection
    associated with session_id (FR-6.1).
    """
    paths = get_session_paths(session_id, data_root=data_root)

    # 1. Delete session directory tree from disk
    if paths["root_dir"].exists():
        try:
            shutil.rmtree(paths["root_dir"])
            logger.info(f"Deleted directory tree for session '{session_id}'")
        except Exception as e:
            logger.warning(f"Error deleting session directory {paths['root_dir']}: {e}")

    # 2. Delete Chroma collection for session
    collection_name = sanitize_collection_name(session_id)
    if HAS_CHROMADB:
        try:
            client = _create_chroma_client(storage_dir)
            if client:
                client.delete_collection(name=collection_name)
                logger.info(f"Deleted Chroma collection '{collection_name}' for session '{session_id}'")
        except Exception as e:
            logger.debug(f"Chroma collection deletion error for {collection_name}: {e}")

    # 3. Clear In-memory fallback store
    InMemoryVectorStore.clear_collection(session_id)
