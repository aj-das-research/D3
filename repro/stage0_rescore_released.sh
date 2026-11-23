#!/usr/bin/env bash
# Stage 0: rescore the authors' released static outputs (single runs, pre-manifest, CPU only) and record them.
# These rows are superseded by regenerated GPU runs in later stages (results.csv keeps the history).
set -euo pipefail
cd "$(dirname "$0")/.."; source env.sh
$D3_PY analysis/saved_output_analysis.py > $D3_OUT/stage0/saved_output_analysis.log
node analysis/error_structure.js > $D3_OUT/stage0/error_structure.log
R="$D3_PY repro/record.py --run_tag rescore_released --notes released-output_single-run"
S=results/error_structure/rescored_summary.csv
O=results/error_structure/orthogonality_summary.csv
Q=results/error_structure/orthogonality_records.csv
CA=results/saved_output_analysis/component_ablation.csv
C1="node analysis/error_structure.js"; C2="python analysis/saved_output_analysis.py"
for f in same_family_n50:same50 diff_family_n50:diff50 diff_family_n300:diff300; do
  file=heterogeneous_${f%%:*}.jsonl; id=${f##*:}
  $R --id T1.$id.rho   --csv $S --where file=$file --col spearman_rho_revised --command "$C1"
  $R --id T1.$id.meanD --csv $S --where file=$file --col mean_D_revised --command "$C1"
  $R --id T1.$id.hard  --csv $S --where file=$file --col hard_case_rate_revised --command "$C1"
done
W="--where source_file=heterogeneous_diff_family_n300.jsonl"
$R --id T2.accR.it0 --csv $O $W --col R_accuracy --command "$C1"
$R --id T2.accD.it0 --csv $O $W --col D_accuracy --command "$C1"
$R --id T2.accC.it0 --csv $O $W --col C_accuracy --command "$C1"
$R --id T2.eceR.it0 --csv $O $W --col R_ECE --command "$C1"
$R --id T2.jsreg.it0 --csv $O $W --col mean_pairwise_JS --command "$C1"
$R --id T2.rho.it0 --csv $S --where file=heterogeneous_diff_family_n300.jsonl --col spearman_rho_revised --command "$C1"
$R --id A.agent.accR --csv $O $W --col R_accuracy --command "$C1"
$R --id A.agent.accD --csv $O $W --col D_accuracy --command "$C1"
$R --id A.agent.accC --csv $O $W --col C_accuracy --command "$C1"
$R --id A.agent.eceR --csv $O $W --col R_ECE --command "$C1"
$R --id A.agent.eceC --csv $O $W --col C_ECE --command "$C1"
$R --id A.agent.jacRD --csv $O $W --col RD_error_jaccard --command "$C1"
$R --id A.agent.uniqR --csv $O $W --col R_only_correct --command "$C1"
$R --id A.agent.uniqD --csv $O $W --col D_only_correct --command "$C1"
$R --id A.agent.uniqC --csv $O $W --col C_only_correct --command "$C1"
$R --id T11.q11840.D --csv $Q --where source_file=heterogeneous_diff_family_n300.jsonl question_id=11840 --col D_revised --command "$C1"
$R --id T11.q12235.D --csv $Q --where source_file=heterogeneous_diff_family_n300.jsonl question_id=12235 --col D_revised --command "$C1"
$R --id A.sig.same50 --csv $CA --where file=experiments/static_disagreement/outputs/heterogeneous_same_family_n50.jsonl schema=per_agent score=D_full --col permutation_p --command "$C2"
$D3_PY repro/build_tracker.py
