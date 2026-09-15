import pytest
from unittest.mock import MagicMock
from pipeline.faithfulness import (
    check_answer_groundedness,
    verify_and_refine_answer,
    UNSUPPORTED_FALLBACK_TEXT,
    FaithfulnessResult
)
from pipeline.chat import DraftAnswer
from llm.client import LLMClient
from graph.build_graph import build_graph, LucidocState

def test_check_answer_groundedness_supported(monkeypatch):
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = '{"verdict": "supported", "reasoning": "Direct match in text"}'

    context = [{"source_filename": "doc.md", "heading_path": "Root", "text": "Python 3.11 is supported."}]
    res = check_answer_groundedness("Python 3.11 is supported.", context, mock_llm)

    assert res.verdict == "supported"

def test_check_answer_groundedness_unsupported(monkeypatch):
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate.return_value = '{"verdict": "unsupported", "reasoning": "Text mentions C++ not Python"}'

    context = [{"source_filename": "doc.md", "heading_path": "Root", "text": "C++ 20 is supported."}]
    res = check_answer_groundedness("Python 3.11 is supported.", context, mock_llm)

    assert res.verdict == "unsupported"

def test_verify_and_refine_two_unsupported_attempts_refusal(monkeypatch):
    """
    FR-5.4 Test: Verifies that when 2 consecutive answers are judged unsupported,
    the exact verbatim fallback refusal is returned.
    """
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)

    mock_llm = MagicMock(spec=LLMClient)
    # Both audit calls return unsupported
    mock_llm.generate.side_effect = [
        '{"verdict": "unsupported", "reasoning": "Claim 1 unsupported"}',  # 1st check
        'Fabricated strict answer text',                                     # 1st retry generation
        '{"verdict": "unsupported", "reasoning": "Retry claim unsupported"}'  # 2nd check
    ]

    draft = DraftAnswer(
        answer_text="Fabricated answer text",
        citations=[{"source_file": "doc.md", "heading_path": "Root", "chunk_id": "c1"}],
        retrieved_context=[{"source_filename": "doc.md", "heading_path": "Root", "text": "Some text"}],
        question="What is X?"
    )

    final = verify_and_refine_answer("test_session", "What is X?", draft, llm_client=mock_llm)

    assert final.answer_text == UNSUPPORTED_FALLBACK_TEXT
    assert final.answer_text == "This isn't covered in the material you provided."
    assert len(final.citations) == 0

def test_graph_faithfulness_chat_integration(monkeypatch, tmp_path):
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")

    md_path = tmp_path / "faith_test.md"
    md_path.write_text("# Overview\n\nLucidoc Faithfulness Test.", encoding="utf-8")

    graph = build_graph()
    state: LucidocState = {
        "uploaded_files": [str(md_path)],
        "session_id": "faithfulness_graph_session",
        "chat_history": [
            {"role": "user", "content": "Tell me about Lucidoc."}
        ]
    }

    result = graph.invoke(state)

    history = result.get("chat_history", [])
    assert len(history) == 2
    assert history[1]["role"] == "assistant"
