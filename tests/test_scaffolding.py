import os
import pytest
from unittest.mock import MagicMock
from graph.build_graph import build_graph, LucidocState
from llm.client import LLMClient

def test_graph_smoke_test(monkeypatch):
    """
    Smoke test running the 5-node LangGraph pipeline with LUCIDOC_MOCK_LLM=1.
    Ensures state passes through all 5 passthrough nodes cleanly without network calls.
    """
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")
    
    app_graph = build_graph()
    initial_state: LucidocState = {
        "uploaded_files": ["doc1.md", "doc2.pdf"],
        "session_id": "test_session_smoke",
        "chat_history": []
    }

    result = app_graph.invoke(initial_state)

    assert result["uploaded_files"] == ["doc1.md", "doc2.pdf"]
    assert result["session_id"] == "test_session_smoke"
    assert "chat_history" in result

def test_llm_client_mock_mode(monkeypatch):
    """
    Verifies that LUCIDOC_MOCK_LLM=1 returns a canned response with zero API calls.
    """
    monkeypatch.setenv("LUCIDOC_MOCK_LLM", "1")
    client = LLMClient()
    
    response = client.generate("Test prompt for mock mode")
    assert response.startswith("[MOCK_RESPONSE]")
    assert client.network_call_count == 0

def test_llm_client_caching(monkeypatch, tmp_path):
    """
    Verifies disk caching behavior: calling generate() twice with identical prompt
    hits the cache on the second call and invokes the underlying API only once.
    """
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)
    
    cache_dir = tmp_path / "llm_cache"
    client = LLMClient(cache_dir=str(cache_dir))
    
    mock_raw_api = MagicMock(return_value="Original API Response")
    monkeypatch.setattr(client, "_raw_api_call", mock_raw_api)

    prompt = "What is the capital of France?"
    
    # First call: should trigger network API call
    res1 = client.generate(prompt, max_tokens=50)
    assert res1 == "Original API Response"
    assert mock_raw_api.call_count == 1

    # Second call: should hit disk cache
    res2 = client.generate(prompt, max_tokens=50)
    assert res2 == "Original API Response"
    assert mock_raw_api.call_count == 1  # Still 1, network API not called again!

def test_llm_client_rate_limit_retry(monkeypatch, tmp_path):
    """
    Verifies exponential backoff retry logic on rate-limit (429) errors.
    """
    monkeypatch.delenv("LUCIDOC_MOCK_LLM", raising=False)
    
    cache_dir = tmp_path / "llm_cache_retry"
    client = LLMClient(cache_dir=str(cache_dir), max_retries=2, initial_backoff=0.01)

    calls = [0]
    def mock_flaky_api(prompt, system_prompt, params):
        calls[0] += 1
        if calls[0] < 2:
            raise Exception("Rate limit hit (429 Too Many Requests)")
        return "Recovered Response"

    monkeypatch.setattr(client, "_raw_api_call", mock_flaky_api)

    res = client.generate("Retry test prompt")
    assert res == "Recovered Response"
    assert calls[0] == 2
