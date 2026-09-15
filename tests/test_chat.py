import pytest
from unittest.mock import MagicMock
from pipeline.index import build_index
from pipeline.chat import answer_question, extract_citations_from_context, format_context_for_prompt
from llm.client import LLMClient
from graph.build_graph import build_graph, LucidocState

def test_extract_citations():
    chunks = [
        {"chunk_id": "doc1.md::0", "source_filename": "doc1.md", "heading_path": "Setup", "text": "Content 1"},
        {"chunk_id": "doc1.md::1", "source_filename": "doc1.md", "heading_path": "Setup", "text": "Content 2"},
        {"chunk_id": "doc2.pdf::0", "source_filename": "doc2.pdf", "heading_path": "API", "text": "Content 3"}
    ]
    citations = extract_citations_from_context(chunks, "Generated answer")
    assert len(citations) == 2
    assert citations[0]["source_file"] == "doc1.md"
    assert citations[1]["source_file"] == "doc2.pdf"

def test_answer_question_mocked_llm(monkeypatch, tmp_path):
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)

    session_id = "chat_test_session"
    chunks = [
        {"chunk_id": "guide.md::0", "text": "Lucidoc uses LangGraph for graph execution.", "source_filename": "guide.md", "heading_path": "Architecture", "chunk_index": 0, "label": "concept"}
    ]
    build_index(session_id, chunks)

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = "Lucidoc uses LangGraph for pipeline orchestration [Source: guide.md > Architecture]."

    draft = answer_question(session_id, "What framework does Lucidoc use?", llm_client=mock_llm)

    assert "LangGraph" in draft.answer_text
    assert len(draft.citations) == 1
    assert draft.citations[0]["source_file"] == "guide.md"
    assert draft.citations[0]["heading_path"] == "Architecture"

def test_graph_chat_node_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")

    md_path = tmp_path / "chat_doc.md"
    md_path.write_text("# Setup Guide\n\nRun pip install to start.", encoding="utf-8")

    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [str(md_path)],
        "session_id": "chat_graph_session",
        "chat_history": [
            {"role": "user", "content": "How do I install Lucidoc?"}
        ]
    }

    result = graph.invoke(state)

    history = result.get("chat_history", [])
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert "content" in history[1]
    assert "citations" in history[1]
