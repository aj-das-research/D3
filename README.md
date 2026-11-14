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
configs/paper.json            all hyperparameters from the paper              (App. D.9)
ddd/                          core method
  disagreement_scorer.py        composite score D(q) and label-free D_LF      (Sec. 3.4)
  hard_case_buffer.py           rolling hard-case buffer, per-agent sampling  (Sec. 3.5, App. D.6)
  diversity_regularizer.py      hinge on mean pairwise JS, NumPy and PyTorch  (Sec. 3.6, Eq. 9)
  coevolution.py                closed co-evolution loop, backend-agnostic    (Sec. 3.6, App. D.6-D.8)
  coevolution_loop.py           standalone batch builder from a predictions file
  split_guard.py                fail-closed train/validation/test checks      (Sec. 4, App. F.1)
  config.py                     loads configs/paper.json
agents/                       agent construction                              (Sec. 3.2, App. D.1-D.3)
  agent_r/                      Agent-R: SFT, then GRPO refinement (+ PPO baseline notebook)
  agent_d/                      Agent-D: domain-adaptive training (Qwen or Mistral base)
  agent_c/                      Agent-C: cross-entropy + ECE loss, temperature scaling
data/
  prepare_mmlu_pro.py           MMLU-Pro hard subset and structural proxy s(q) (Sec. 4, App. C)
  create_split_manifest.py      2,416 / 300 / 300 category-stratified split
  mmlu_pro_hard_subset.{json,csv}
  splits/                       split manifest and balance summary
  pool/                         public model-pool response matrix and validated difficulty targets (Table 4)
experiments/
  static_disagreement/          pre-co-evolution (Iteration 0) runs and saved outputs (Table 1)
  coevolution/                  GPU co-evolution runner and Hugging Face backend (Table 2, Fig. 2, Tables 8-9, App. G.7)
  self_consistency/             reasoning-path sampler for the self-consistency baseline (Sec. 6.5, App. G.5)
evaluation/
  build_pool_matrix.py          response matrix from public MMLU-Pro model predictions
  static_validation.py          saved static outputs against the validated targets
  difficulty_targets.py         1 - pass rate and 2PL IRT difficulty from a model-pool response matrix (Table 4)
  baselines.py                  verbalised confidence, single-model entropy, self-consistency, composite D_LF
  compare_estimators.py         Spearman against validated targets, mean +/- sd over seeds (Table 4)
  downstream.py                 evaluation triage, selective prediction, compute allocation (Table 5)
analysis/
  saved_output_analysis.py      canonical rescoring, component ablation, bootstrap CIs, permutation tests
  error_structure.js            per-agent accuracy, ECE, error Jaccard, unique-correct counts
  plot_figures.py               Figures 2, 3 and 5
results/                      tables and figures regenerated from the saved outputs
scripts/                      seed, ablation and sweep drivers
tests/                        unit tests (no model downloads)
```

## What this release contains

| Paper item | Where |
| --- | --- |
| Structural complexity proxy `s(q)`, hard subset (Sec. 4, App. C) | `data/` |
| Split manifest and guards (App. F.1) | `data/splits/`, `ddd/split_guard.py` |
| Composite score `D(q)`, label-free `D_LF` (Sec. 3.4) | `ddd/disagreement_scorer.py` |
| Hard-case buffer and sampling rules (Sec. 3.5, App. D.6) | `ddd/coevolution.py`, `ddd/hard_case_buffer.py` |
| Diversity regulariser (Eq. 9) | `ddd/diversity_regularizer.py` |
| Agent training (Sec. 3.2, App. D.1-D.3) | `agents/` |
| Static disagreement (Table 1) | `experiments/static_disagreement/`, with saved outputs |
| Co-evolution, regulariser and agent ablations (Table 2, Fig. 2, Tables 8-9) | `experiments/coevolution/`, `scripts/` |
| Component ablation and significance tests (Table 3, App. G.8) | `analysis/saved_output_analysis.py` (also reads co-evolution predictions) |
| Validated targets and estimator comparison (Table 4) | `evaluation/` |
| Downstream utility (Table 5, App. I) | `evaluation/downstream.py` |
| Self-consistency sweep (App. G.5), buffer and threshold sweeps (App. G.7) | `experiments/self_consistency/`, `scripts/` |
| Per-agent diagnostics (Figs. 3-4, Table 11) | `analysis/error_structure.js` |
| Figures 2, 3 and 5 | `analysis/plot_figures.py` |

Saved outputs are included only for the static runs (see
[Saved outputs](#saved-outputs)). The co-evolution, Table 4-5 and sweep
scripts implement the procedures specified in the paper; their run outputs
are not part of this release, so these scripts have not been checked against
the reported numbers. The exploratory design-space study (App. H) is not
included. MUSE and EIP, the two external estimators in Table 4, are run with
their official implementations:
[LARK-NLP-Lab/MUSE](https://github.com/LARK-NLP-Lab/MUSE) and
[Leozz04/EIP](https://github.com/Leozz04/EIP).

## Trained adapters

The LoRA adapters used for the static runs are hosted on Kaggle:

| Agent | Base model | Download |
| --- | --- | --- |
| Agent-R, SFT stage | Qwen2.5-3B-Instruct | [agent-r-sft-adapter](https://www.kaggle.com/datasets/ebadshabbir/agent-r-sft-adapter) |
| Agent-R, after GRPO | Qwen2.5-3B-Instruct | [agent-r-v0-grpo-adapter-zip](https://www.kaggle.com/datasets/ebadshabbir/agent-r-v0-grpo-adapter-zip) |
| Agent-D | Qwen2.5-3B-Instruct | [agent-d-result](https://www.kaggle.com/datasets/ebadshabbir/agent-d-result) (adapter in `agent_d_v0/`) |
| Agent-C | Qwen2.5-3B-Instruct | [agent-c-result](https://www.kaggle.com/datasets/ebadshabbir/agent-c-result) (adapter and `temperature.json` in `agent_c_v0/`) |

These are the same-family adapters. The Mistral-7B Agent-D adapter used for
the different-family setting and the co-evolved adapters are not released.

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
python -m pytest tests                       # unit tests, CPU only
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
# Agent-R: SFT on GSM8K, then GRPO refinement on top of the SFT adapter
BASE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct OUTPUT_DIR=outputs/agent_r_sft \
  python agents/agent_r/train_agent_r_sft.py
BASE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct SFT_ADAPTER_PATH=outputs/agent_r_sft \
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

## Co-evolution (Table 2, Figure 2, Tables 8-9)

`ddd/coevolution.py` implements the closed loop of Section 3.6. Each of 30
iterations scores a fresh batch of training questions with the current
adapters, adds questions at or above the batch's 80th percentile of `D(q)` to a
500-question FIFO buffer, exposes the top 200, lets each agent draw its own
slice with the Appendix D.6 rules, and takes 50 LoRA steps per agent on

```
CE(gold)  [+ lambda_ece * ECE loss for Agent-C]  +  lambda_div * mean_q max(0, tau_JS - Jbar(q))
```

with the other agents' distributions held fixed. All agents are then
re-scored on the validation split; the test split is scored once at the end.
`s(q)` is used only for the logged correlations.

```bash
python experiments/coevolution/run_coevolution.py \
  --base_model /path/to/Qwen2.5-3B-Instruct \
  --agent_d_base_model /path/to/Mistral-7B-Instruct-v0.3 \
  --agent_r /path/to/agent_r --agent_d /path/to/agent_d_mistral --agent_c /path/to/agent_c \
  --seed 42 --output_dir outputs/coevolution/reg/seed42 --eval_test --save_adapters
```

Omit `--agent_d_base_model` for the same-family setting. Flags for the
ablations: `--no_regularizer` (Table 8), `--agents R,C` (Table 9),
`--buffer_size` and `--hard_percentile` (Appendix G.7). The drivers in
`scripts/` run all three seeds with and without the regulariser, and the
ablation sweeps. Each run writes `iterations.jsonl` (per-iteration accuracy,
ECE, mean JS, `rho(D, s)` and `rho(D_LF, s)`), `val_predictions.jsonl`, and,
with `--eval_test`, `test_metrics.json` and `test_predictions.jsonl`.
Hyperparameters come from `configs/paper.json` (`tau_JS = 0.30`,
`lambda_div = 1.0`, learning rate `1e-5`).

`ddd/coevolution_loop.py` is a lighter tool that builds per-agent hard
batches from an existing predictions file and hands each batch to an update
command of your own.

## Validated targets and downstream utility (Tables 4-5)

`data/pool/mmlu_pro_hard_pool_responses.csv` records, for each of the 3,016
hard-subset questions, whether each of 47 public models answered it
correctly. It is built from the per-question predictions that the MMLU-Pro
authors publish in the `eval_results/` folder of
[TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro); one model
covering less than 90% of the subset is left out. `data/pool/difficulty_targets.csv`
holds the targets derived from it (1 - pass rate and 2PL IRT difficulty).
To rebuild both:

```bash
python evaluation/build_pool_matrix.py --download_dir pool_zips --out data/pool/mmlu_pro_hard_pool_responses.csv
python evaluation/difficulty_targets.py --responses data/pool/mmlu_pro_hard_pool_responses.csv \
  --out data/pool/difficulty_targets.csv
```

This public pool is not necessarily the pool used for Table 4.
`evaluation/static_validation.py` scores the saved static outputs against
these targets.

To evaluate new runs:

```bash
# 1. Targets from any model-pool response matrix (question_id + one 0/1 column per model)
python evaluation/difficulty_targets.py --responses pool_responses.csv --out results/targets.csv

# 2. Per-question scores for each method
python evaluation/baselines.py --method composite --input outputs/coevolution/reg/seed42/test_predictions.jsonl \
  --seed 42 --out results/scores/dlf_coevolved.csv
python evaluation/baselines.py --method entropy --input single_model.jsonl --out results/scores/entropy.csv

# 3. Table 4
python evaluation/compare_estimators.py --targets results/targets.csv \
  --estimator "D_LF co-evolved=results/scores/dlf_coevolved.csv" \
  --estimator "Single-model entropy=results/scores/entropy.csv" --out results/table4.csv

# 4. Table 5
python evaluation/downstream.py --score "D_LF co-evolved=results/scores/dlf_coevolved.csv" \
  --score "Random=random" --pool pool_responses.csv --committee committee_correct.csv \
  --deliberation deliberation.csv --out results/table5.csv
```

The input formats are documented at the top of each script. The 2PL fit is a
joint maximum-a-posteriori estimate with weak Gaussian priors.

## Self-consistency sweep (Appendix G.5)

```bash
MODEL=/path/to/Qwen2.5-3B-Instruct scripts/run_self_consistency_sweep.sh
```

This samples 100 reasoning paths per test question and scores budgets of 5,
10, 20, 40 and 100 with the same composite, treating each third of the
samples as one pseudo-agent.

## Figures

```bash
python analysis/plot_figures.py --coevolution_dir outputs/coevolution
```

Figures 3 and 5 are drawn from the saved static outputs; Figure 2 needs
co-evolution runs.

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
