#!/bin/zsh
# Loop-count sweep on the variable-loop research checkpoint.
# For each pass count: make a model dir (symlinked weights, config with that
# num_passes), serve it on :8081, run strategy V on all problems, stop.
#
#   ./sweep_loops.sh            # default order 3 5 2 4
#   ./sweep_loops.sh 1 6        # other pass counts
set -u
cd "${0:A:h}"
BASE=~/models/TML-d24-varloop-mlx
PORT=8081
PASSES=($@)
(( ${#PASSES} )) || PASSES=(3 5 2 4)

for n in $PASSES; do
  dir=~/models/varloop-p$n
  mkdir -p $dir
  for f in $BASE/*; do
    [[ ${f:t} == config.json ]] || ln -sf $f $dir/${f:t}
  done
  python3 - "$BASE/config.json" "$dir/config.json" "$n" <<'EOF'
import json, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(src))
t = c.get("text_config", c)
t["loop_config"]["num_passes"] = n
json.dump(c, open(dst, "w"), indent=2)
print(f"{dst}: num_passes={n}")
EOF

  python3 serve.py --model $dir --port $PORT --prompt-cache-bytes 4000000000 \
    --decode-concurrency 4 --prompt-concurrency 4 >> results/server_$PORT.log 2>&1 &
  server=$!
  until curl -s -m 2 -o /dev/null http://127.0.0.1:$PORT/v1/models; do
    kill -0 $server 2>/dev/null || { echo "P$n: server failed to start"; exit 1; }
    sleep 5
  done
  echo "P$n: server up (pid $server)"

  HF_HUB_OFFLINE=1 ~/.local/share/uv/tools/mlx-lm/bin/python -m orchestra.run V --model $dir --url http://127.0.0.1:$PORT/v1 --tag P$n 2>&1 | grep --line-buffered -v Warning
  python3 -m orchestra.report

  kill $server; pkill -f "mlx_lm.server --model $dir"; wait $server 2>/dev/null
done
