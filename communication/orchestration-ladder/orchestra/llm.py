"""Client for the local Inkling server, with an on-disk cache and usage metering.

Every call is cached by (messages, parameters, sample index), so a rerun of a
strategy, or a new strategy that shares a prefix of calls, costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = os.environ.get("INKLING_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("INKLING_MODEL", "pipenetwork/Inkling-2-Mini-MLX-bf16")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"
TIMEOUT_S = 3600

# Global cap on in-flight requests: the server batches ~4 well on one GPU.
_slots = threading.Semaphore(int(os.environ.get("INKLING_CONCURRENCY", "4")))


@dataclass
class Reply:
    content: str
    reasoning: str
    completion_tokens: int
    prompt_tokens: int
    seconds: float
    cached: bool
    finish_reason: str


@dataclass
class LLM:
    """One per problem, so usage is metered per problem."""

    temperature: float = 0.7
    calls: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    model_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def chat(
        self,
        messages: list[dict],
        *,
        effort: str = "high",
        max_tokens: int = 8192,
        sample: int = 0,
    ) -> Reply:
        params = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"reasoning_effort": effort},
        }
        key = hashlib.sha256(
            json.dumps({**params, "sample": sample}, sort_keys=True).encode()
        ).hexdigest()
        path = CACHE_DIR / key[:2] / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            reply = Reply(**{**data, "cached": True})
        else:
            reply = self._request(params)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**reply.__dict__, "cached": False}))
            tmp.replace(path)
        with self._lock:
            self.calls += 1
            self.completion_tokens += reply.completion_tokens
            self.prompt_tokens += reply.prompt_tokens
            self.model_seconds += reply.seconds
        return reply

    def chat_many(self, messages: list[dict], n: int, **kw) -> list[Reply]:
        """n independent samples of the same messages, in parallel."""
        with ThreadPoolExecutor(n) as ex:
            return list(ex.map(lambda i: self.chat(messages, sample=i, **kw), range(n)))

    def _request(self, params: dict) -> Reply:
        body = json.dumps(params).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", body, {"Content-Type": "application/json"}
        )
        last_err = None
        for attempt in range(4):
            with _slots:
                start = time.monotonic()
                try:
                    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                        data = json.load(r)
                except Exception as e:  # server busy / reset: back off and retry
                    last_err = e
                    time.sleep(5 * (attempt + 1))
                    continue
                seconds = time.monotonic() - start
            choice = data["choices"][0]
            msg = choice["message"]
            usage = data.get("usage", {})
            return Reply(
                content=msg.get("content") or "",
                reasoning=msg.get("reasoning") or "",
                completion_tokens=usage.get("completion_tokens", 0),
                prompt_tokens=usage.get("prompt_tokens", 0),
                seconds=seconds,
                cached=False,
                finish_reason=choice.get("finish_reason") or "",
            )
        raise RuntimeError(f"LLM request failed after retries: {last_err}")
