#!/usr/bin/env bash
# Bare model server vs the gateway, back to back, same shape as Week 2: all users arrive at once (n = c),
# 100 tokens, the same prompt. Appends one labelled JSON line per run to $OUT.
#
# Expects both servers already running:  model server on $BARE_URL, gateway (in front of it) on $GATEWAY_URL.
# Run from week03-gateway/:
#   OUT=results/llama8b-4090.jsonl LABEL="Llama 3.1 8B, RTX 4090, RunPod" ./scripts/bench_matrix.sh
#
# Overrides for a quick rehearsal:  C_LIST="1 4"  REPEATS=0
set -euo pipefail

BARE_URL=${BARE_URL:-http://127.0.0.1:8001}
GATEWAY_URL=${GATEWAY_URL:-http://127.0.0.1:8080}
OUT=${OUT:-results/bench.jsonl}
LABEL=${LABEL:-unlabelled}
C_LIST=${C_LIST:-"1 4 16 64"}
REPEATS=${REPEATS:-2}            # extra runs at c=16, because answer lengths vary between runs
PROMPT="Write a long story about a robot."

mkdir -p "$(dirname "$OUT")"

# A header line saying exactly what was measured, so the file explains itself later.
python3 - "$OUT" "$LABEL" "$C_LIST" "$REPEATS" <<'EOF'
import json, sys, datetime
out, label, c_list, repeats = sys.argv[1:]
meta = {"_about": label, "shape": "n=c (all at once), max_tokens=100, prompt 'Write a long story about a robot.'",
        "c_list": c_list, "c16_repeats": int(repeats), "started": datetime.datetime.now().isoformat(timespec="seconds")}
open(out, "a").write(json.dumps(meta) + "\n")
EOF

run() {  # run <target> <url> <c> <run number>
  local target=$1 url=$2 c=$3 n_run=$4
  echo "=== $target  c=$c  run $n_run"
  # --timeout 900: at c=64 the last user on the bare server waits behind 63 answers (several minutes on a GPU).
  local summary
  summary=$(uv run python scripts/loadtest.py /v1/chat/completions --prompt "$PROMPT" --max-tokens 100 \
              -n "$c" -c "$c" --timeout 900 --base-url "$url" --json | tee /dev/stderr | tail -1)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(target=sys.argv[2], run=int(sys.argv[3])); print(json.dumps(d))' \
    "$summary" "$target" "$n_run" >> "$OUT"
}

for c in $C_LIST; do
  run bare "$BARE_URL" "$c" 1
  run gateway "$GATEWAY_URL" "$c" 1
done

for r in $(seq 2 $((REPEATS + 1))); do
  [ "$REPEATS" -gt 0 ] || break
  run bare "$BARE_URL" 16 "$r"
  run gateway "$GATEWAY_URL" 16 "$r"
done

echo
echo "Done. Results in $OUT. Gateway counters:"
curl -s "$GATEWAY_URL/health" 2>/dev/null || python3 -c "import urllib.request as u; print(u.urlopen('$GATEWAY_URL/health').read().decode())"
echo
