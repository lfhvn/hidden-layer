#!/bin/zsh
# After the loop sweep ends, rescue the best P run (highest strict pass rate,
# ties broken by hard passes) on :8081 with orchestra.rescue (F16 and X32).
#
#   ./rescue_best_p.sh        # pick the best P automatically
#   ./rescue_best_p.sh 4      # rescue P4
set -u
cd "${0:A:h}"
PORT=8081
PY=~/.local/share/uv/tools/mlx-lm/bin/python

while pgrep -f sweep_loops.sh > /dev/null; do sleep 60; done

n=${1:-$(python3 - <<'EOF'
import json, pathlib
best = None
for f in sorted(pathlib.Path("results").glob("P*.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    key = (sum(r["passed"] for r in rows) / len(rows),
           sum(r["passed"] for r in rows if r["difficulty"] == "hard"))
    print(f"  {f.stem}: pass {key[0]:.0%}, hard {key[1]}", file=__import__("sys").stderr)
    if best is None or key > best[0]:
        best = (key, f.stem[1:])
print(best[1])
EOF
)}
dir=~/models/varloop-p$n
echo "rescuing P$n ($dir)"

python3 serve.py --model $dir --port $PORT --prompt-cache-bytes 4000000000 \
  --decode-concurrency 4 --prompt-concurrency 4 >> results/server_$PORT.log 2>&1 &
server=$!
until curl -s -m 2 -o /dev/null http://127.0.0.1:$PORT/v1/models; do
  kill -0 $server 2>/dev/null || { echo "P$n: server failed to start"; exit 1; }
  sleep 5
done
echo "P$n: server up (pid $server)"

HF_HUB_OFFLINE=1 $PY -m orchestra.rescue P$n --model $dir --url http://127.0.0.1:$PORT/v1 2>&1 \
  | grep --line-buffered -v Warning
python3 -m orchestra.rescue --report

kill $server; pkill -f "mlx_lm.server --model $dir"; wait $server 2>/dev/null
