"""Client for the local Inkling server, with an on-disk cache and usage metering.

Every call is cached by (messages, parameters, sample index), so a rerun of a
strategy, or a new strategy that shares a prefix of calls, costs nothing.

Budget forcing: when a reply hits max_tokens while still thinking (no answer),
chat() closes the thinking and makes the model answer in up to ANSWER_TOKENS
more tokens. This needs the model's chat template, so the harness runs under
the mlx-lm Python (see run_all.sh).
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
COMPLETE_CACHE_DIR = CACHE_DIR.parent / "complete"
FORCE = os.environ.get("INKLING_FORCE", "1") != "0"
ANSWER_TOKENS = 4096
THINK = "<|message_model|><|content_thinking|>"
END = "<|end_message|>"
ANSWER = END + "<|message_model|><|content_text|>"
NUDGE = (
    "\n\nI am out of thinking time. I will stop here and write my best complete "
    "solution now."
)
# A 16k-token answer can take over an hour when the GPU is shared; a client
# timeout makes the harness retry from scratch while the server keeps going.
TIMEOUT_S = int(os.environ.get("INKLING_TIMEOUT", 6 * 3600))

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
    forced: bool = False  # the answer was forced after the thinking hit max_tokens


@dataclass
class LLM:
    """One per problem, so usage is metered per problem."""

    temperature: float = 0.7
    calls: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    model_seconds: float = 0.0
    forced: int = 0
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
        if FORCE and reply.finish_reason == "length" and not reply.content and reply.reasoning:
            reply = self._force(messages, effort, reply)
        with self._lock:
            self.calls += 1
            self.completion_tokens += reply.completion_tokens
            self.prompt_tokens += reply.prompt_tokens
            self.model_seconds += reply.seconds
            self.forced += reply.forced
        return reply

    def _force(self, messages: list[dict], effort: str, reply: Reply) -> Reply:
        """Resume after the capped thinking, close it, and get the answer."""
        head = _tokenizer().apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, reasoning_effort=effort
        )
        a = complete(head + THINK + reply.reasoning + NUDGE + ANSWER, ANSWER_TOKENS, self.temperature)
        return Reply(
            content=a["text"],
            reasoning=reply.reasoning,
            completion_tokens=reply.completion_tokens + a["completion_tokens"],
            prompt_tokens=reply.prompt_tokens,
            seconds=reply.seconds + a["seconds"],
            cached=reply.cached,
            finish_reason=a["finish_reason"],
            forced=True,
        )

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
        # Up to ~6 minutes of retries, so a server restart (serve.py) is
        # ridden out instead of failing the problem.
        for attempt in range(8):
            with _slots:
                start = time.monotonic()
                try:
                    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                        data = json.load(r)
                except Exception as e:  # server busy / reset: back off and retry
                    last_err = e
                    time.sleep(min(10 * (attempt + 1), 60))
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


_tok_lock = threading.Lock()
_toks: dict = {}


def _tokenizer():
    with _tok_lock:
        if MODEL not in _toks:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from mlx_lm.utils import _download, load_tokenizer

            _toks[MODEL] = load_tokenizer(_download(os.path.expanduser(MODEL)))
        return _toks[MODEL]


def complete(prompt: str, max_tokens: int, temperature: float = 0.7) -> dict:
    """Raw text completion that stops at the end of the current message. Cached."""
    params = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": [END],
    }
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = COMPLETE_CACHE_DIR / key[:2] / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(
        f"{BASE_URL}/completions", json.dumps(params).encode(), {"Content-Type": "application/json"}
    )
    last_err = None
    for attempt in range(8):
        with _slots:
            start = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    data = json.load(r)
            except Exception as e:  # server busy / restarting: back off and retry
                last_err = e
                time.sleep(min(10 * (attempt + 1), 60))
                continue
        choice, usage = data["choices"][0], data.get("usage", {})
        out = {
            "text": choice.get("text") or "",
            "finish_reason": choice.get("finish_reason") or "",
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "seconds": round(time.monotonic() - start, 1),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(path)
        return out
    raise RuntimeError(f"completion failed after retries: {last_err}")
