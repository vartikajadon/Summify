import pytest
from pipeline.ingest import ingest_files
from pipeline.chunk import chunk_doc, chunk_files, Chunk
from graph.build_graph import build_graph, LucidocState

def test_chunk_structured_doc(tmp_path):
    """
    Verifies that a document with clear heading structure is split along headings,
    and each chunk inherits the correct heading_path hierarchy.
    """
    md_path = tmp_path / "structured.md"
    md_path.write_text(
        "# System Architecture\n\nOverview of the system architecture.\n\n"
        "## Database Layer\n\nChroma DB is used for vector storage.\n\n"
        "### Collections\n\nEach session gets a isolated collection.\n\n"
        "## API Layer\n\nFastAPI handles REST endpoints.",
        encoding="utf-8"
    )

    ingested = ingest_files([str(md_path)])
    assert len(ingested) == 1

    chunks = chunk_doc(ingested[0].to_dict())
    assert len(chunks) >= 4

    paths = [c.heading_path for c in chunks]
    assert "System Architecture" in paths[0]
    assert "System Architecture > Database Layer" in paths[1]
    assert "System Architecture > Database Layer > Collections" in paths[2]
    assert "System Architecture > API Layer" in paths[3]

def test_chunk_unstructured_large_paragraph(tmp_path):
    """
    Verifies that a giant unstructured paragraph without headings is split into bounded chunks.
    """
    large_text = "Word " * 2500  # ~12,500 characters
    txt_path = tmp_path / "large_blob.txt"
    txt_path.write_text(large_text, encoding="utf-8")

    ingested = ingest_files([str(txt_path)])
    chunks = chunk_doc(ingested[0].to_dict())

    assert len(chunks) > 1
    max_allowed_chars = 500 * 4 + 100  # max_tokens=500 -> ~2000 chars plus small tolerance
    for c in chunks:
        assert len(c.text) <= max_allowed_chars
        assert c.source_filename == "large_blob.txt"

def test_chunk_traceability(tmp_path):
    """
    Verifies that all chunks maintain source filename, chunk_index, and valid IDs.
    """
    md_path = tmp_path / "trace.md"
    md_path.write_text("# Title\n\nContent A\n\n## Section 1\n\nContent B", encoding="utf-8")

    ingested = ingest_files([str(md_path)])
    chunks = chunk_doc(ingested[0].to_dict())

    for idx, c in enumerate(chunks):
        assert c.source_filename == "trace.md"
        assert c.chunk_index == idx
        assert c.chunk_id == f"trace.md::{idx}"
        assert isinstance(c.heading_path, str)

def test_graph_chunking_integration(tmp_path):
    """
    Verifies end-to-end graph state flow through ingest and chunk_classify nodes.
    """
    md_path = tmp_path / "graph_test.md"
    md_path.write_text("# Graph Test\n\nTesting graph state chunking.", encoding="utf-8")

    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [str(md_path)],
        "session_id": "chunk_graph_session"
    }

    result = graph.invoke(state)

    raw_docs = result.get("raw_text", [])
    chunks = result.get("chunks", [])

    assert len(raw_docs) == 1
    assert len(chunks) >= 1
    assert chunks[0]["source_filename"] == "graph_test.md"
    assert "Graph Test" in chunks[0]["heading_path"]
