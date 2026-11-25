"""Tests for the local (OpenAI-compatible) provider, against a stub server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from harness import llm_call
from harness.defaults import BASELINE_MODEL


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.server.body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(self.server.status)
            self.end_headers()
            self.wfile.write(json.dumps(self.server.reply).encode())

        def log_message(self, *args):
            pass

    s = HTTPServer(("127.0.0.1", 0), Handler)
    s.status = 200
    s.reply = {
        "choices": [{"message": {"content": "4", "reasoning": "2+2"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    s.url = f"http://127.0.0.1:{s.server_port}/v1"
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s
    s.shutdown()
    s.server_close()


def call(server, **kwargs):
    kwargs.setdefault("model", BASELINE_MODEL)
    return llm_call("2+2?", provider="local", base_url=server.url, **kwargs)


def test_request_and_response(server):
    r = call(server, system_prompt="Be brief.", reasoning_effort="high", seed=1, num_ctx=99)
    assert (r.text, r.tokens_in, r.tokens_out) == ("4", 7, 3)
    assert r.metadata == {"finish_reason": "stop", "reasoning": "2+2"}
    assert [m["role"] for m in server.body["messages"]] == ["system", "user"]
    assert server.body["model"] == BASELINE_MODEL
    assert server.body["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert server.body["seed"] == 1
    assert "num_ctx" not in server.body


def test_warns_when_thinking_uses_the_budget(server):
    server.reply["choices"][0] = {"message": {"content": "", "reasoning": "..."}, "finish_reason": "length"}
    with pytest.warns(UserWarning, match="before answering"):
        assert call(server).text == ""


def test_server_error_raises(server):
    server.status = 500
    with pytest.raises(RuntimeError, match="returned 500"):
        call(server)


def test_unreachable_server_raises():
    with pytest.raises(RuntimeError, match="No local model server"):
        llm_call("hi", provider="local", model="m", base_url="http://127.0.0.1:1/v1")


def test_baseline_is_not_sent_to_other_providers():
    with pytest.raises(ValueError, match="No model given"):
        llm_call("hi", provider="anthropic")
