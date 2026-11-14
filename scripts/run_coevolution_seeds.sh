#!/usr/bin/env bash
# Co-evolution over the paper's three seeds, with and without the regulariser
# (Table 2, Table 8, Figure 2). Set the model and adapter paths below.
set -euo pipefail
BASE=${BASE:?set BASE to the Qwen2.5-3B-Instruct path}
D_BASE=${D_BASE:-}            # Mistral-7B-Instruct-v0.3 path for the different-family setting
AGENT_R=${AGENT_R:?}; AGENT_D=${AGENT_D:?}; AGENT_C=${AGENT_C:?}
OUT=${OUT:-outputs/coevolution}
EXTRA=()
[ -n "$D_BASE" ] && EXTRA+=(--agent_d_base_model "$D_BASE")
for SEED in 42 1337 2024; do
  python experiments/coevolution/run_coevolution.py --base_model "$BASE" "${EXTRA[@]}" \
    --agent_r "$AGENT_R" --agent_d "$AGENT_D" --agent_c "$AGENT_C" \
    --seed "$SEED" --output_dir "$OUT/reg/seed$SEED" --eval_test --save_adapters
  python experiments/coevolution/run_coevolution.py --base_model "$BASE" "${EXTRA[@]}" \
    --agent_r "$AGENT_R" --agent_d "$AGENT_D" --agent_c "$AGENT_C" \
    --seed "$SEED" --output_dir "$OUT/noreg/seed$SEED" --no_regularizer --eval_test
done
python analysis/plot_figures.py --coevolution_dir "$OUT"
