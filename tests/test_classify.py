import pytest
from unittest.mock import MagicMock
from pipeline.classify import classify_chunk_zeroshot, classify_chunks, parse_label_from_response, VALID_TAXONOMY
from llm.client import LLMClient
from graph.build_graph import build_graph, LucidocState

def test_parse_label_from_response():
    assert parse_label_from_response('{"label": "setup/install", "reasoning": "Install instructions"}') == "setup/install"
    assert parse_label_from_response('```json\n{"label": "api-reference"}\n```') == "api-reference"
    assert parse_label_from_response('Invalid JSON output string') == "other"
    assert parse_label_from_response('{"label": "unknown_category"}') == "other"

def test_classify_chunk_zeroshot_mocked(monkeypatch):
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)
    
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = '{"label": "setup/install", "reasoning": "Pip command found"}'

    chunk_dict = {
        "chunk_id": "test.md::0",
        "text": "Run pip install -r requirements.txt to set up the environment.",
        "source_filename": "test.md",
        "heading_path": "Installation",
        "chunk_index": 0
    }

    result = classify_chunk_zeroshot(chunk_dict, mock_llm)

    assert result.label == "setup/install"
    assert result.method == "zero-shot"
    assert result.chunk_id == "test.md::0"
    assert result.label in VALID_TAXONOMY

def test_classify_chunk_zeroshot_malformed_fallback(monkeypatch):
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = "Sorry, I cannot classify this text chunk!"

    chunk_dict = {
        "chunk_id": "test.md::1",
        "text": "Some text",
        "source_filename": "test.md",
        "heading_path": "Root",
        "chunk_index": 1
    }

    result = classify_chunk_zeroshot(chunk_dict, mock_llm)
    assert result.label == "other"

def test_graph_classification_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")
    
    md_path = tmp_path / "class_test.md"
    md_path.write_text("## Setup\n\nRun pip install.", encoding="utf-8")

    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [str(md_path)],
        "session_id": "classify_graph_session"
    }

    result = graph.invoke(state)

    classified = result.get("classified_chunks", [])
    assert len(classified) >= 1
    assert classified[0]["method"] == "zero-shot"
    assert classified[0]["label"] in VALID_TAXONOMY
