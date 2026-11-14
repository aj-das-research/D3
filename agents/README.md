# Agents

The three agents differ in training objective, not in prompt (paper Section
3.2; hyperparameters in Appendix D.1-D.2). All use LoRA adapters; base weights
stay frozen. The scripts were written for Kaggle GPU sessions and default to
`/kaggle/input` and `/kaggle/working` paths; every path can be overridden.
Model weights and adapter archives are not stored in this repository.

## Agent-R: reasoning specialist

- `agent_r/train_agent_r_sft.py`: supervised fine-tuning with a LoRA adapter
  on GSM8K worked solutions. Its output is the SFT adapter used below.
- `agent_r/train_agent_r_grpo.py` and `agent_r/agent_r_grpo_training.ipynb`:
  GRPO refinement of the Agent-R SFT adapter on GSM8K, with a reward that
  combines answer-format validity and correctness.
- `agent_r/agent_r_ppo_baseline.ipynb`: the PPO variant compared against GRPO
  in Appendix D.3.

Inputs are read from the environment:

```bash
# 1. SFT
BASE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct OUTPUT_DIR=outputs/agent_r_sft \
  python agents/agent_r/train_agent_r_sft.py

# 2. GRPO on top of the SFT adapter
BASE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct SFT_ADAPTER_PATH=outputs/agent_r_sft \
  python agents/agent_r/train_agent_r_grpo.py
```

The released SFT run used GSM8K only; Appendix D.1 lists GSM8K and MATH.

## Agent-D: domain specialist

- `agent_d/train_agent_d.py` and `agent_d/agent_d_training.ipynb`: two phases
  on one adapter, a domain-adaptive causal-LM pass on PubMed abstracts followed
  by SFT on MedQA and PubMedQA.

```bash
python agents/agent_d/train_agent_d.py \
  --model_path /path/to/base-model \
  --output_dir outputs/agent_d
```

Pass a Qwen2.5-3B-Instruct base for the same-family setting or
Mistral-7B-Instruct-v0.3 for the different-family setting.

## Agent-C: calibration specialist

- `agent_c/train_agent_c.py`: trains a multiple-choice scorer with
  `loss = CE + ece_weight * ECE` (Eq. 1) on the training split, then fits one
  scalar temperature on the validation split and saves it as
  `temperature.json` beside the adapter.

```bash
python agents/agent_c/train_agent_c.py \
  --model_path /path/to/Qwen2.5-3B-Instruct \
  --mmlu_json data/mmlu_pro_hard_subset.json \
  --split_manifest data/splits/mmlu_pro_hard_split_manifest.csv \
  --output_dir outputs/agent_c
```

At inference Agent-C scores next-token logits over the option letters `A-J`,
applies `softmax(logits / T)` with the saved temperature, and returns the
arg-max letter with its probability as confidence.
