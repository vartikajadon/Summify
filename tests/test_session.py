import pytest
from pathlib import Path
from session.manager import create_session, export_all, delete_session, get_session_paths
from pipeline.structure import assemble_document
from pipeline.index import build_index, query_index

def test_session_lifecycle(tmp_path):
    data_root = str(tmp_path / "data")
    
    # 1. Create Session
    session_id = create_session(data_root=data_root)
    assert len(session_id) > 0

    paths = get_session_paths(session_id, data_root=data_root)
    assert paths["uploads_dir"].exists()
    assert paths["exports_dir"].exists()

    # 2. Export All
    chunks = [
        {"chunk_id": "c1", "text": "Session text sample", "source_filename": "f.md", "heading_path": "Heading", "chunk_index": 0, "label": "concept"}
    ]
    doc = assemble_document(chunks, title="Session Lifecycle Doc")

    exports = export_all(session_id, structured_doc=doc, data_root=data_root)

    assert "markdown" in exports
    assert "docx" in exports
    assert "pdf" in exports
    assert len(exports["markdown"]) > 0
    assert len(exports["docx"]) > 100
    assert len(exports["pdf"]) > 50

    assert (paths["exports_dir"] / "documentation.md").exists()
    assert (paths["exports_dir"] / "documentation.docx").exists()
    assert (paths["exports_dir"] / "documentation.pdf").exists()

def test_delete_session_complete_cleanup(tmp_path):
    """
    FR-6.1 Test: Verifies that after delete_session, no files remain on disk
    and the vector collection is completely dropped.
    """
    data_root = str(tmp_path / "data")
    storage_dir = str(tmp_path / "chroma_db")
    session_id = create_session(data_root=data_root)
    paths = get_session_paths(session_id, data_root=data_root)

    # Seed index and files
    chunks = [
        {"chunk_id": "c1", "text": "Content to delete", "source_filename": "del.md", "heading_path": "Root", "chunk_index": 0, "label": "concept"}
    ]
    build_index(session_id, chunks, storage_dir=storage_dir)

    # Delete session
    delete_session(session_id, data_root=data_root, storage_dir=storage_dir)

    # Assert directory tree is gone
    assert not paths["root_dir"].exists()

    # Assert vector index query returns nothing
    query_res = query_index(session_id, "delete", storage_dir=storage_dir)
    assert len(query_res) == 0
