# Local patch: tool call parser for Inkling 2 (not part of upstream mlx-lm).

"""Tool call parser for Inkling 2.

A tool call is one message:

    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>

The header (``<|message_model|>NAME``) is dropped by the response format in
``mlx_lm.chat_templates.inkling2``; this parser gets only the JSON body.
"""

import json
from typing import Any, Optional

tool_call_start = "<|content_invoke_tool_json|>"
tool_call_end = "<|end_message|>"


def parse_tool_call(text: str, tools: Optional[Any] = None) -> dict[str, Any]:
    body = text.strip()
    if body.endswith(tool_call_end):
        body = body[: -len(tool_call_end)].rstrip()
    call = json.loads(body)
    name = call.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Inkling tool call has no name.")
    arguments = call.get("args", call.get("arguments", {}))
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    return {"name": name, "arguments": arguments or {}}
