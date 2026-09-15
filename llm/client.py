import hashlib
import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

import threading
import random

_groq_lock = threading.Lock()
_last_groq_time = 0.0

class LLMClient:
    """
    Unified LLM Client wrapper for Lucidoc.
    Supports disk caching, dry-run mock mode, request rate throttling, and backoff retry with model fallback on 429 rate limit errors.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        cache_dir: Optional[str] = None,
        max_retries: int = 6,
        initial_backoff: float = 1.5,
    ):
        self.config = self._load_config(config_path)
        self.provider = provider or self.config.get("provider", "groq")
        self.model = model or self.config.get("model", "openai/gpt-oss-20b")
        self.max_tokens = max_tokens or self.config.get("max_tokens", 1024)
        
        storage_cfg = self.config.get("storage", {})
        self.cache_dir = Path(cache_dir or storage_cfg.get("cache_dir", "./.cache/llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.network_call_count = 0  # Useful for testing cache verification

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}

    def is_mock_mode(self) -> bool:
        return os.environ.get("LUCIDOC_MOCK_LLM") == "1"

    def _compute_cache_key(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        payload = {
            "prompt": prompt,
            "model": model,
            "provider": self.provider,
            "params": params,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.debug(f"LLM cache hit: {cache_key}")
                    return data.get("response")
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        return None

    def _save_cached_response(self, cache_key: str, response: str, prompt: str, params: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "cache_key": cache_key,
                    "prompt": prompt,
                    "model": self.model,
                    "provider": self.provider,
                    "params": params,
                    "response": response,
                    "timestamp": time.time(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate completion for a prompt.
        Checks mock mode first, then disk cache, then calls LLM API with exponential backoff on 429.
        """
        params = {"max_tokens": kwargs.get("max_tokens", self.max_tokens), "temperature": kwargs.get("temperature", 0.0)}
        if system_prompt:
            params["system_prompt"] = system_prompt

        # 1. Check Mock Mode
        if self.is_mock_mode():
            logger.info("LUCIDOC_MOCK_LLM=1 is active; returning mock response.")
            prompt_combined = f"{system_prompt or ''} {prompt}".lower()
            if "verdict" in prompt_combined or "faithfulness" in prompt_combined:
                return '{"verdict": "supported", "reasoning": "Mock LLM mode auto-approved verdict"}'
            elif "label" in prompt_combined or "classifier" in prompt_combined:
                return '{"label": "concept", "reasoning": "Mock LLM mode auto-classified"}'
            return f"[MOCK_RESPONSE] Responded to prompt length {len(prompt)}"

        # 2. Check Disk Cache
        cache_key = self._compute_cache_key(prompt, self.model, params)
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            return cached

        # 3. Network Call with Exponential Backoff
        response_text = self._execute_with_retry(prompt, system_prompt, params)
        
        # Clean Qwen thinking tags if present
        if response_text and "<think>" in response_text and "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()

        # 4. Save to Cache
        self._save_cached_response(cache_key, response_text, prompt, params)
        return response_text

    def _execute_with_retry(self, prompt: str, system_prompt: Optional[str], params: Dict[str, Any]) -> str:
        backoff = self.initial_backoff
        last_exception = None
        fallback_model = "openai/gpt-oss-20b"

        for attempt in range(self.max_retries + 1):
            try:
                self.network_call_count += 1
                return self._raw_api_call(prompt, system_prompt, params)
            except Exception as e:
                last_exception = e
                err_msg = str(e)
                is_rate_limit = any(k in err_msg.lower() for k in ["429", "413", "rate limit", "itpm", "tpm"])
                is_404 = "404" in err_msg or "model_not_found" in err_msg.lower()

                if is_404:
                    if self.provider == "groq" and self.model != fallback_model:
                        logger.warning(f"Model {self.model} returned 404. Falling back to active model {fallback_model}...")
                        old_model = self.model
                        try:
                            self.model = fallback_model
                            return self._raw_api_call(prompt, system_prompt, params)
                        except Exception as fb_err:
                            logger.error(f"Fallback model call failed: {fb_err}")
                            raise fb_err
                        finally:
                            self.model = old_model
                    else:
                        raise e

                if is_rate_limit:
                    # Truncate prompt on 413/ITPM overflow
                    if "413" in err_msg or "itpm" in err_msg.lower():
                        prompt = prompt[:2500]

                    if attempt < self.max_retries:
                        sleep_time = backoff
                        import re
                        match = re.search(r"try again in ([\d\.]+)s", err_msg, re.IGNORECASE)
                        if match:
                            try:
                                parsed_sec = float(match.group(1)) + 1.0
                                sleep_time = max(sleep_time, parsed_sec)
                            except Exception:
                                pass

                        sleep_time = sleep_time + random.uniform(0.2, 1.5)
                        logger.warning(f"Rate/Token limit (429/413) encountered. Waiting {sleep_time:.1f}s before retry (attempt {attempt + 1}/{self.max_retries})...")
                        time.sleep(sleep_time)
                        backoff = min(backoff * 2.0, 30.0)
                    else:
                        if self.provider == "groq" and self.model != fallback_model:
                            logger.warning(f"Model {self.model} rate limited after {self.max_retries} attempts. Falling back to {fallback_model}...")
                            old_model = self.model
                            try:
                                self.model = fallback_model
                                return self._raw_api_call(prompt, system_prompt, params)
                            except Exception as fb_err:
                                logger.error(f"Fallback model call failed: {fb_err}")
                                raise fb_err
                            finally:
                                self.model = old_model
                        logger.error(f"LLM API call failed: {e}")
                        raise e
                else:
                    logger.error(f"LLM API call failed: {e}")
                    raise e

        raise last_exception or RuntimeError("LLM call failed after retries")

    def _raw_api_call(self, prompt: str, system_prompt: Optional[str], params: Dict[str, Any]) -> str:
        """
        Invokes actual provider SDK (Groq or Ollama).
        """
        if self.provider == "groq":
            return self._call_groq(prompt, system_prompt, params)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt, params)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_groq(self, prompt: str, system_prompt: Optional[str], params: Dict[str, Any]) -> str:
        global _last_groq_time
        try:
            import groq
        except ImportError:
            raise ImportError("groq package is required for provider='groq'. Install via requirements.txt.")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        with _groq_lock:
            now = time.time()
            elapsed = now - _last_groq_time
            if elapsed < 1.5:
                time.sleep(1.5 - elapsed)
            _last_groq_time = time.time()

        client = groq.Groq(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=params.get("max_tokens", self.max_tokens),
            temperature=params.get("temperature", 0.0),
        )
        return response.choices[0].message.content or ""

    def _call_ollama(self, prompt: str, system_prompt: Optional[str], params: Dict[str, Any]) -> str:
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package is required for provider='ollama'. Install via requirements.txt.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": params.get("max_tokens", self.max_tokens),
                "temperature": params.get("temperature", 0.0),
            }
        )
        return response.get("message", {}).get("content", "")
