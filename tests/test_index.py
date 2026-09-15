import pytest
from pipeline.index import build_index, add_to_index, remove_from_index, query_index
from graph.build_graph import build_graph, LucidocState

@pytest.fixture
def temp_chroma_dir(tmp_path):
    return str(tmp_path / "chroma_db")

def test_strict_session_isolation(temp_chroma_dir):
    """
    CRITICAL TEST (FR-4.2): Asserts that querying Session A never leaks chunks belonging to Session B.
    """
    session_a_chunks = [
        {"chunk_id": "a_1", "text": "Python is a dynamic programming language.", "source_filename": "py.md", "heading_path": "Python", "chunk_index": 0, "label": "concept"},
        {"chunk_id": "a_2", "text": "Install Python using pyenv or python.org.", "source_filename": "py.md", "heading_path": "Install", "chunk_index": 1, "label": "setup/install"}
    ]

    session_b_chunks = [
        {"chunk_id": "b_1", "text": "Java runs on the JVM virtual machine.", "source_filename": "java.md", "heading_path": "Java", "chunk_index": 0, "label": "concept"},
        {"chunk_id": "b_2", "text": "Maven and Gradle build Java projects.", "source_filename": "java.md", "heading_path": "Build", "chunk_index": 1, "label": "setup/install"}
    ]

    # Build indexes for two separate sessions
    coll_a = build_index("session_alpha", session_a_chunks, storage_dir=temp_chroma_dir)
    coll_b = build_index("session_beta", session_b_chunks, storage_dir=temp_chroma_dir)

    assert coll_a != coll_b

    # Query Session Alpha
    results_a = query_index("session_alpha", "programming language", k=5, storage_dir=temp_chroma_dir)
    assert len(results_a) > 0
    for r in results_a:
        assert r["session_id"] == "session_alpha"
        assert "Java" not in r["text"]
        assert "JVM" not in r["text"]

    # Query Session Beta
    results_b = query_index("session_beta", "programming language", k=5, storage_dir=temp_chroma_dir)
    assert len(results_b) > 0
    for r in results_b:
        assert r["session_id"] == "session_beta"
        assert "Python" not in r["text"]

def test_incremental_add_and_remove(temp_chroma_dir):
    session_id = "inc_session"
    initial_chunks = [
        {"chunk_id": "c1", "text": "Initial doc text", "source_filename": "init.md", "heading_path": "Root", "chunk_index": 0, "label": "concept"}
    ]

    build_index(session_id, initial_chunks, storage_dir=temp_chroma_dir)

    res1 = query_index(session_id, "doc", k=5, storage_dir=temp_chroma_dir)
    assert len(res1) == 1

    # Incremental add
    new_chunks = [
        {"chunk_id": "c2", "text": "Secondary added text", "source_filename": "added.md", "heading_path": "Root", "chunk_index": 0, "label": "concept"}
    ]
    add_to_index(session_id, new_chunks, storage_dir=temp_chroma_dir)

    res2 = query_index(session_id, "text", k=5, storage_dir=temp_chroma_dir)
    assert len(res2) == 2

    # Incremental remove
    remove_from_index(session_id, chunk_ids=["c1"], storage_dir=temp_chroma_dir)
    res3 = query_index(session_id, "text", k=5, storage_dir=temp_chroma_dir)
    assert len(res3) == 1
    assert res3[0]["chunk_id"] == "c2"

def test_graph_index_node_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")

    md_path = tmp_path / "idx_test.md"
    md_path.write_text("# Index Test\n\nVector index test content.", encoding="utf-8")

    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [str(md_path)],
        "session_id": "idx_graph_session"
    }

    result = graph.invoke(state)

    ref = result.get("vector_index_ref", "")
    assert ref.startswith("session_idx_graph_session")
