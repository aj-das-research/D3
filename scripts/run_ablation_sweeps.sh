#!/usr/bin/env bash
# Agent-removal ablation (Table 9) and buffer/threshold sweeps (Appendix G.7), seed 42.
set -euo pipefail
BASE=${BASE:?}; AGENT_R=${AGENT_R:?}; AGENT_D=${AGENT_D:?}; AGENT_C=${AGENT_C:?}
D_BASE=${D_BASE:-}; OUT=${OUT:-outputs/ablations}
EXTRA=(); [ -n "$D_BASE" ] && EXTRA+=(--agent_d_base_model "$D_BASE")
run() { python experiments/coevolution/run_coevolution.py --base_model "$BASE" "${EXTRA[@]}" \
          --agent_r "$AGENT_R" --agent_d "$AGENT_D" --agent_c "$AGENT_C" --seed 42 --eval_test "$@"; }
for AGENTS in D,C R,C R,D; do run --agents "$AGENTS" --output_dir "$OUT/agents_${AGENTS/,/}"; done
for B in 50 100 200 500 1000 2000 5000; do run --buffer_size "$B" --output_dir "$OUT/buffer_$B"; done
for P in 50 60 70 80 90 95; do run --hard_percentile "$P" --output_dir "$OUT/percentile_$P"; done
