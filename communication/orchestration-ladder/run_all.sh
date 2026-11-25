#!/bin/sh
# Run the ladder in the order that maximizes cache reuse. The mlx-lm Python
# renders the chat template that budget forcing needs (orchestra/llm.py).
cd "$(dirname "$0")"
PY=~/.local/share/uv/tools/mlx-lm/bin/python
for s in ${@:-S0 S2 S3 S1 S4 S5}; do
  HF_HUB_OFFLINE=1 $PY -m orchestra.run "$s" 2>&1 | grep --line-buffered -v Warning
  python3 -m orchestra.report
done
