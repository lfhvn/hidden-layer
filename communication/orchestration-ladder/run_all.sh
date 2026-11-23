#!/bin/sh
# Run the ladder in the order that maximizes cache reuse.
cd "$(dirname "$0")"
for s in S0 S2 S3 S1 S4 S5; do
  python3 -m orchestra.run "$s" || exit 1
  python3 -m orchestra.report
done
