#!/usr/bin/env bash
# Self-consistency sample-budget sweep (Appendix G.5).
set -euo pipefail
MODEL=${MODEL:?set MODEL to the Qwen2.5-3B-Instruct path}
OUT=${OUT:-outputs/self_consistency}
python experiments/self_consistency/sample_answers.py --model "$MODEL" --split test --num_samples 100 \
  --out "$OUT/samples_test.jsonl"
for B in 5 10 20 40 100; do
  python evaluation/baselines.py --method self_consistency --budget "$B" \
    --input "$OUT/samples_test.jsonl" --out "$OUT/scores_sc$B.csv"
done
