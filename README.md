# Disagreement, Diversity, and Difficulty in Heterogeneous Multi-Agent Language Models

Code, data splits, and saved per-question outputs for the paper
**"Disagreement, Diversity, and Difficulty in Heterogeneous Multi-Agent
Language Models"** (Findings of EACL 2027).

The paper asks when disagreement among language-model agents says something
about how hard a question is. Three agents that differ in *training signal*
(a GRPO-refined reasoner, a biomedical specialist, and a calibration-trained
analyst) answer each multiple-choice question; a composite probabilistic
score reads their disagreement; high-disagreement questions feed a
co-evolution loop in which a Jensen-Shannon floor keeps the agents from
collapsing into one behaviour.

## Repository layout

```
ddd/                          core method
  disagreement_scorer.py        composite score D(q) and label-free D_LF   (Sec. 3.4)
  hard_case_buffer.py           rolling hard-case buffer, per-agent sampling (Sec. 3.5, App. D.6)
  diversity_regularizer.py      hinge penalty on mean pairwise JS            (Sec. 3.6, Eq. 9)
  coevolution_loop.py           per-agent hard-batch builder and update driver (Sec. 3.6)
  split_guard.py                fail-closed train/validation/test checks     (Sec. 4, App. F.1)
agents/                       agent construction                             (Sec. 3.2, App. D.1-D.3)
  agent_r/                      Agent-R: GRPO refinement (+ PPO baseline notebook)
  agent_d/                      Agent-D: domain-adaptive training
  agent_c/                      Agent-C: cross-entropy + ECE loss, temperature scaling
data/
  prepare_mmlu_pro.py           MMLU-Pro hard subset and structural proxy s(q) (Sec. 4, App. C)
  create_split_manifest.py      2,416 / 300 / 300 category-stratified split
  mmlu_pro_hard_subset.{json,csv}
  splits/                       split manifest and balance summary
experiments/static_disagreement/   pre-co-evolution (Iteration 0) runs        (Sec. 5-6.1, Table 1)
  run_homogeneous.py            three role-prompted copies of one hosted model
  run_heterogeneous.py          same-family R/D/C adapters on Qwen2.5-3B-Instruct
  run_heterogeneous_multibackbone.py   Qwen R/C + Mistral-7B D
  outputs/                      saved per-question outputs
analysis/
  saved_output_analysis.py      canonical rescoring, component ablation, bootstrap CIs, permutation tests
  error_structure.js            per-agent accuracy, ECE, error Jaccard, unique-correct counts
results/                      tables regenerated from the saved outputs
tests/                        unit tests (no model downloads)
scripts/download_qwen_model.py
```

## What this release contains

| Paper item | Where | Status |
| --- | --- | --- |
| Structural complexity proxy `s(q)`, hard subset (Sec. 4, App. C) | `data/` | Code and prepared data |
| Split manifest and guards (App. F.1) | `data/splits/`, `ddd/split_guard.py` | Code and manifest |
| Composite score `D(q)`, label-free `D_LF` (Sec. 3.4) | `ddd/disagreement_scorer.py` | Code and tests |
| Hard-case buffer and sampling rules (Sec. 3.5, App. D.6) | `ddd/hard_case_buffer.py` | Code |
| Diversity regulariser (Eq. 9) | `ddd/diversity_regularizer.py` | Code and tests (NumPy report + PyTorch loss) |
| Agent-R GRPO, Agent-D, Agent-C training (Sec. 3.2) | `agents/` | Code |
| Static disagreement, Table 1 | `experiments/static_disagreement/` | Runners and saved outputs for the heterogeneous settings (see [Saved outputs](#saved-outputs)) |
| Per-agent diagnostics, Figures 3-4, Table 11 | `analysis/error_structure.js`, `results/error_structure/` | Code and regenerated tables |
| Significance tests (App. G.8) | `analysis/saved_output_analysis.py` | Code, applied to the saved static outputs |
| Co-evolution batches (Sec. 3.6) | `ddd/coevolution_loop.py` | Batch builder and update driver; adapter-update commands are supplied by the user |

Not included in this release: trained LoRA adapters, the Agent-R supervised
fine-tuning stage that precedes GRPO, per-iteration co-evolution outputs
(Table 2, Figure 2, Tables 8-9), the validated-target and downstream-utility
evaluations (Tables 4-5), the self-consistency and control sweeps
(App. G.5-G.7), the exploratory design-space study (App. H), and the
figure-plotting code.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/EbadShabbir/Multi_agent_neurips.git
cd Multi_agent_neurips
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests and the analysis scripts need only `numpy`, `scipy`, `pandas` and
`pytest`. Training and the heterogeneous runners need a CUDA GPU (the paper
used a single 16 GB T4). The homogeneous runner calls a hosted
OpenAI-compatible API: copy `.env.example` to `.env` and set the key for the
backend you use.

## Quick start without a GPU

```bash
python -m pytest tests                       # 20 unit tests
python analysis/saved_output_analysis.py     # writes results/saved_output_analysis/
node analysis/error_structure.js             # writes results/error_structure/
```

Both analysis commands read `experiments/static_disagreement/outputs/` and
perform no model inference. `component_ablation.csv` reports, for every saved
file and score variant, Spearman correlation with the structural proxy, a
bootstrap 95% interval (1,000 question resamples) and a permutation p-value
(10,000 permutations).

## Data and splits

`data/mmlu_pro_hard_subset.{json,csv}` hold the 3,016 MMLU-Pro records with the
highest structural-proxy score; `difficulty_score` is `s(q)` from Eq. 10.
To rebuild them from the Hugging Face dataset:

```bash
python data/prepare_mmlu_pro.py
```

The split manifest assigns every question to `train` (2,416), `validation`
(300) or `test` (300), stratified by category and proxy quintile with seed
2026. Rebuilding it is deterministic:

```bash
python data/create_split_manifest.py
# SHA-256 d06f8afb50f7fff4fb0f5dc0a7ca19c040d25dbe0fad3c8544d1fd6353129ed8
```

Agent-C training, the static runners and the co-evolution driver load the
manifest through `ddd/split_guard.py`, which aborts on unknown or duplicate
question IDs and, when a manifest SHA-256 is passed, on a hash mismatch.

## Training the agents

See [`agents/README.md`](agents/README.md) for inputs and outputs. In short:

```bash
# Agent-R: GRPO refinement on top of an SFT adapter
BASE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct SFT_ADAPTER_PATH=/path/to/agent_r_sft \
  python agents/agent_r/train_agent_r_grpo.py

# Agent-D: domain-adaptive LM + SFT on MedQA and PubMedQA
python agents/agent_d/train_agent_d.py --model_path /path/to/base-model \
  --output_dir outputs/agent_d

# Agent-C: CE + ECE loss on the training split, temperature fitted on validation
python agents/agent_c/train_agent_c.py --model_path /path/to/Qwen2.5-3B-Instruct \
  --mmlu_json data/mmlu_pro_hard_subset.json \
  --split_manifest data/splits/mmlu_pro_hard_split_manifest.csv \
  --output_dir outputs/agent_c
```

## Static disagreement (Table 1)

```bash
# Homogeneous role-prompted baseline (hosted Llama-3.3-70B by default)
python experiments/static_disagreement/run_homogeneous.py --eval-split test --limit 30

# Same-family heterogeneous: three Qwen2.5-3B-Instruct adapters
python experiments/static_disagreement/run_heterogeneous.py \
  --base_model /path/to/Qwen2.5-3B-Instruct \
  --agent_r /path/to/agent_r --agent_d /path/to/agent_d --agent_c /path/to/agent_c \
  --eval_split test --output_dir outputs/static

# Different-family heterogeneous: Qwen R/C + Mistral-7B D
python experiments/static_disagreement/run_heterogeneous_multibackbone.py \
  --qwen_base_model /path/to/Qwen2.5-3B-Instruct \
  --mistral_base_model /path/to/Mistral-7B-Instruct-v0.3 \
  --agent_r /path/to/agent_r --agent_c /path/to/agent_c --agent_d /path/to/agent_d_mistral \
  --eval_split test --output_dir outputs/static
```

The runners default to the `validation` split; `--eval_split test` must be
requested explicitly. Each run writes one JSONL row per question (answers,
confidences and, for the heterogeneous runners, full option distributions);
the heterogeneous runners also write a run summary. The adaptive 80th-percentile hard-case threshold and the
score ablations are applied afterwards by `analysis/saved_output_analysis.py`.

## Co-evolution

```bash
python ddd/coevolution_loop.py \
  --predictions_jsonl /path/to/heterogeneous_train_predictions.jsonl \
  --max_iterations 30 --buffer_size 500 --sample_size 200 \
  --min_js_divergence 0.30 --diversity_weight 1.0 \
  --agent_r_cmd "python my_update_r.py --hard_batch {batch} --iteration {iteration}" \
  --agent_d_cmd "python my_update_d.py --hard_batch {batch} --iteration {iteration}" \
  --agent_c_cmd "python my_update_c.py --hard_batch {batch} --iteration {iteration}"
```

The driver keeps only training-split rows, rebuilds the hard-case buffer,
writes one hard batch per agent under `iter_XX/`, and runs the update command
for each agent (`--dry_run` builds the batches only). Every batch row carries
the question's pairwise divergences and a `regularizer_hint` with the current
penalty, and `ddd/diversity_regularizer.py` provides `torch_diversity_loss` for use inside
an update script:

```python
from diversity_regularizer import torch_diversity_loss

loss = task_loss + torch_diversity_loss(
    {"R": r_choice_logits, "D": d_choice_logits, "C": c_choice_logits},
    weight=1.0, min_js_divergence=0.30,
)
```

The command-line and function defaults for the regulariser (`0.05` and
`0.03`) are development values; the paper uses `lambda_div = 1.0` and
`tau_JS = 0.30`, passed explicitly above. Adapter-update scripts and
re-scoring between iterations are not part of this release.

## Saved outputs

`experiments/static_disagreement/outputs/` holds one saved run per setting.
Running the two analysis commands above gives:

| Saved file | n | Spearman rho(D, s) | Mean D | Hard-case rate |
| --- | ---: | ---: | ---: | ---: |
| `heterogeneous_same_family_n50.jsonl` | 50 | 0.154 | 0.417 | 0.200 |
| `heterogeneous_diff_family_n50.jsonl` | 50 | -0.008 | 0.478 | 0.200 |
| `heterogeneous_diff_family_n300.jsonl` | 300 | 0.003 | 0.475 | 0.203 |

On the different-family n = 300 file, `error_structure.js` gives per-agent
accuracy 0.253 / 0.227 / 0.473 (R / D / C), ECE 0.348 / 0.083 / 0.052,
pairwise error Jaccard 0.640 (R-D), 0.585 (R-C), 0.512 (D-C), unique-correct
counts 15 / 24 / 61, and mean pairwise JS 0.317.

Notes on provenance:

- Each file is a single run; the repository does not contain per-seed outputs.
- These runs were produced before the split manifest was introduced and are
  drawn from the first records of `data/mmlu_pro_hard_subset.csv`, which span
  all three manifest splits. Use the runners with `--eval_split test` for a
  sealed-split evaluation.
- `homogeneous_n109.jsonl` and `homogeneous_n300.jsonl` are homogeneous runs
  on 109 and 300 questions. They are not the n = 30 run reported in Table 1,
  which is not among the saved outputs. Under the canonical scorer they give
  rho = 0.286 and 0.344; see `results/saved_output_analysis/component_ablation.csv`.

More detail is in
[`experiments/static_disagreement/outputs/README.md`](experiments/static_disagreement/outputs/README.md).

## License and acknowledgements

Code is released under the [MIT License](LICENSE). The prepared question files
are derived from [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
(Wang et al., 2024), which its authors distribute under the MIT license; please
cite MMLU-Pro when using them.
