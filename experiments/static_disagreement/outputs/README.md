# Saved static outputs

One JSONL row per question. These are the saved runs behind the static
(pre-co-evolution) analysis; `analysis/saved_output_analysis.py` and
`analysis/error_structure.js` read this folder directly.

| File | Agents | n |
| --- | --- | ---: |
| `heterogeneous_same_family_n50.jsonl` | Agent-R/D/C adapters on Qwen2.5-3B-Instruct | 50 |
| `heterogeneous_diff_family_n50.jsonl` | Qwen Agent-R/C, Mistral-7B Agent-D | 50 |
| `heterogeneous_diff_family_n300.jsonl` | Qwen Agent-R/C, Mistral-7B Agent-D | 300 |
| `homogeneous_n109.jsonl` | Three role-prompted copies of one hosted model | 109 |
| `homogeneous_n300.jsonl` | Three role-prompted copies of one hosted model | 300 |

## Schemas

Heterogeneous rows (`per_agent`): for each of `R`, `D`, `C` the selected
letter, its confidence, and `all_probs`, the full distribution over the option
letters. Homogeneous rows (`per_role`): for each role the selected letter and a
verbalised confidence; distributions are reconstructed by the maximum-entropy
rule of Appendix D.4.

Every row also carries `question_id`, `category`, `correct_answer` and
`difficulty_score` (the structural proxy without the category adjustment;
see the main README). The `D_q`, `D_semantic` and
`is_hard_case` fields were written at run time with a fixed 0.5 threshold; the
analysis scripts recompute the composite score from the agent outputs and
apply the adaptive 80th-percentile threshold.

The `*.run_summary.json` files are the summaries written by the runners at the
end of each heterogeneous run, using the run-time score and the fixed
threshold.

## Provenance

- Each file is a single run. Per-seed outputs are not included.
- The runs predate the split manifest. Each covers the first *n* records of
  `data/mmlu_pro_hard_subset.csv` (the homogeneous n = 109 file holds 109 of
  the first 165 records), so the questions span the manifest's train,
  validation and test splits: 235 / 34 / 31 for the n = 300 files, 36 / 5 / 9
  for the n = 50 files and 84 / 12 / 13 for the n = 109 file.
- The homogeneous files are not the n = 30 run reported in Table 1 of the
  paper; that run's outputs are not in the repository.
