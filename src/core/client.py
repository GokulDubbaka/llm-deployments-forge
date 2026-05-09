"""
LLMClient -- production-grade client for local and remote inference backends.

Supports:
  - Ollama  (http://localhost:11434)     -- /api/generate, /api/chat
  - vLLM    (http://localhost:8000)      -- /v1/completions, /v1/chat/completions
  - Any OpenAI-compatible endpoint       -- /v1/chat/completions

Features:
  - Connection pooling via requests.Session
  - Exponential backoff retry (configurable attempts)
  - Streaming response aggregation
  - Health check before benchmarking
  - Per-request latency telemetry
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT   = 120   # seconds per request
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMP       = 0.7
RETRY_TOTAL        = 3
RETRY_BACKOFF      = 1.0


# ─── Backend enum ─────────────────────────────────────────────────────────────

class Backend(str, Enum):
    OLLAMA  = "ollama"
    VLLM    = "vllm"
    OPENAI  = "openai"   # Any OpenAI-compatible endpoint

    @classmethod
    def detect(cls, base_url: str) -> "Backend":
        """Auto-detect backend type from URL heuristics."""
        if "11434" in base_url:
            return cls.OLLAMA
        if "8000" in base_url or "vllm" in base_url.lower():
            return cls.VLLM
        return cls.OPENAI


# ─── Response model ───────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    backend: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.latency_sec <= 0:
            return 0.0
        return self.completion_tokens / self.latency_sec


# ─── Core Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified inference client.

    Usage:
        client = LLMClient("http://localhost:11434")  # Auto-detects Ollama
        resp   = client.generate("llama3", "Explain IDOR vulnerabilities.")
        print(resp.text, resp.tokens_per_second)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = RETRY_TOTAL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.timeout  = timeout
        self.backend  = Backend.detect(base_url)

        # Session with connection pooling + retry
        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        self._session.mount("http://",  HTTPAdapter(max_retries=retry))
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

        logger.info("LLMClient initialized: backend=%s url=%s", self.backend, self.base_url)

    # ── Health check ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Ping the backend and confirm it is responsive."""
        try:
            endpoints = {
                Backend.OLLAMA: "/api/tags",
                Backend.VLLM:   "/v1/models",
                Backend.OPENAI: "/v1/models",
            }
            url  = self.base_url + endpoints[self.backend]
            resp = self._session.get(url, timeout=5)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """Return available model names from the backend."""
        try:
            if self.backend == Backend.OLLAMA:
                resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
            else:
                resp = self._session.get(f"{self.base_url}/v1/models", timeout=10)
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception as exc:
            logger.error("list_models failed: %s", exc)
            return []

    # ── Generate (non-streaming) ──────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMP,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        """
        Single-turn completion. Returns full LLMResponse including telemetry.

        Args:
            model:       Model name as returned by list_models().
            prompt:      User prompt string.
            system:      Optional system prompt (supported on all backends).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens:  Maximum completion tokens.

        Returns:
            LLMResponse with text, token counts, and latency.

        Raises:
            requests.HTTPError: On non-2xx responses after retries exhausted.
            RuntimeError:       If backend returns empty or malformed response.
        """
        t0 = time.perf_counter()

        if self.backend == Backend.OLLAMA:
            payload, url = self._build_ollama_payload(model, prompt, system, temperature, max_tokens)
        else:
            payload, url = self._build_openai_payload(model, prompt, system, temperature, max_tokens)

        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()

        latency = time.perf_counter() - t0
        return self._parse_response(resp.json(), model, latency)

    def stream(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMP,
    ) -> Iterator[str]:
        """
        Streaming token-by-token generation.
        Yields text chunks as they arrive from the backend.
        """
        if self.backend == Backend.OLLAMA:
            payload, url = self._build_ollama_payload(model, prompt, system, temperature, stream=True)
        else:
            payload, url = self._build_openai_payload(model, prompt, system, temperature, stream=True)

        import json as _json
        with self._session.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                raw = line.decode("utf-8")
                if raw.startswith("data: "):
                    raw = raw[6:]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(raw)
                    if self.backend == Backend.OLLAMA:
                        yield chunk.get("response", "")
                    else:
                        yield chunk["choices"][0].get("delta", {}).get("content", "")
                except Exception:
                    continue

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_ollama_payload(
        self, model, prompt, system, temperature=DEFAULT_TEMP, max_tokens=DEFAULT_MAX_TOKENS, stream=False
    ):
        url = f"{self.base_url}/api/generate"
        payload: dict = {
            "model":  model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        return payload, url

    def _build_openai_payload(
        self, model, prompt, system, temperature=DEFAULT_TEMP, max_tokens=DEFAULT_MAX_TOKENS, stream=False
    ):
        url = f"{self.base_url}/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      stream,
        }
        return payload, url

    def _parse_response(self, data: dict, model: str, latency: float) -> LLMResponse:
        """Normalize Ollama and OpenAI response schemas into LLMResponse."""
        if self.backend == Backend.OLLAMA:
            text             = data.get("response", "")
            prompt_tokens    = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
        else:
            choices          = data.get("choices", [{}])
            msg              = choices[0].get("message", {}) if choices else {}
            text             = msg.get("content", "")
            usage            = data.get("usage", {})
            prompt_tokens    = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        if not text:
            raise RuntimeError(f"Backend returned empty response: {data}")

        return LLMResponse(
            text=text.strip(),
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            latency_sec=round(latency, 3),
            backend=self.backend.value,
        )

    def __repr__(self) -> str:
        return f"LLMClient(backend={self.backend.value!r}, url={self.base_url!r})"
