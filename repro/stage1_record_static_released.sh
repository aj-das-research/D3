#!/usr/bin/env bash
# Record Table 1 same-family rows from the released Kaggle adapters scored on the sealed test split (repro/score_agents.py).
set -euo pipefail
cd "$(dirname "$0")/.."; source env.sh
R="$D3_PY repro/record.py --run_tag faithful_released_adapters --deviation_id I2"
G="$D3_OUT/static/same_family_released_n50/seed*/metrics.json"
C="python repro/score_agents.py --split test --n 50 --base_model \$QWEN3B --agent R=\$AR_REL --agent D=\$AD_REL --agent C=\$AC_REL --out \$D3_OUT/static/same_family_released_n50"
$R --id T1.same50.rho   --glob "$G" --key rho_D --command "$C"
$R --id T1.same50.meanD --glob "$G" --key mean_D --command "$C"
$R --id T1.same50.hard  --glob "$G" --key hard_rate --command "$C"
