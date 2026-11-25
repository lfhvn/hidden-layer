# Local patch: response format for Inkling 2 (not part of upstream mlx-lm).

"""Response format of Inkling 2.

A turn is a sequence of messages, each with a header and a content tag:

    <|message_model|><|content_thinking|>reasoning<|end_message|>
    <|message_model|><|content_text|>answer<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{...}<|end_message|>
    <|content_model_end_sampling|>

Header text is dropped. The state machine adds the tool -> header edge on
``<|end_message|>`` from the tool parser.

The checkpoint's own Jinja template is still used for prompts; only this
RESPONSE_FORMAT is taken from here.
"""

RESPONSE_FORMAT = {
    "initial_state": "header",
    "transitions": {
        "header": [
            ("<|content_thinking|>", "reasoning"),
            ("<|content_text|>", "normal"),
            ("<|content_invoke_tool_text|>", "normal"),
            ("<|content_tool_error|>", "normal"),
            ("<|content_invoke_tool_json|>", "tool"),
        ],
        "reasoning": [("<|end_message|>", "header")],
        "normal": [("<|end_message|>", "header")],
    },
}
