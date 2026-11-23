#!/usr/bin/env bash
# Launch one co-evolution run in the background on a given GPU (logs to $D3_OUT/logs/<tag>.log).
# usage: repro/launch_coevo.sh GPU TAG [extra run_coevolution.py args...]
#   setting is chosen by env SETTING=same|diff (diff = Mistral-7B Agent-D), adapters by AR/AD/AC env overrides.
set -euo pipefail
cd "$(dirname "$0")/.."; source env.sh; source repro/paths.sh
GPU=$1; TAG=$2; shift 2
AR=${AR:-$AR_REL}; AC=${AC:-$D3_OUT/adapters/agent_c_manifest}
if [ "${SETTING:-diff}" = diff ]; then AD=${AD:-$D3_OUT/adapters/agent_d_mistral}; DB="--agent_d_base_model $MISTRAL7B"; else AD=${AD:-$AD_REL}; DB=""; fi
OUT=$D3_OUT/coevolution/$TAG
mkdir -p $OUT
echo "[launch] $TAG on GPU $GPU -> $OUT" 
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup $D3_PY experiments/coevolution/run_coevolution.py \
  --base_model $QWEN3B $DB --agent_r $AR --agent_d $AD --agent_c $AC --output_dir $OUT --eval_test --save_adapters "$@" \
  > $D3_OUT/logs/coevo_$(echo $TAG | tr / _).log 2>&1 < /dev/null &
