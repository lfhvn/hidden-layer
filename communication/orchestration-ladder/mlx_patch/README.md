# mlx-lm patch for Inkling 2 tool calls

`mlx_lm.server` on the Inkling 2 PR branch (ml-explore/mlx-lm#2086) returns
tool calls and reasoning as raw text. These files make it parse Inkling's
format (`<|content_invoke_tool_json|>{"name":…,"args":…}<|end_message|>`).

Install mlx-lm from the PR, then copy into its `mlx_lm/` package:

| File here | Goes to |
|---|---|
| `tool_parsers_inkling2.py` | `mlx_lm/tool_parsers/inkling2.py` |
| `chat_templates_inkling2.py` | `mlx_lm/chat_templates/inkling2.py` |
| `models_inkling2_init.diff` | append to `mlx_lm/models/inkling2/__init__.py` |

Reinstalling mlx-lm removes the patch.
