"""
Fast Kaggle GRPO pass for Agent-R.

Use this after you already have an Agent-R SFT LoRA adapter.

Recommended Kaggle flow:
1. Add two Kaggle inputs:
   - Qwen2.5-3B-Instruct base model directory
   - Agent-R SFT adapter directory
2. Run the install cell below once, then restart the notebook kernel.
3. Run this script/notebook cell.

Install cell:
    !pip install -q \
      "huggingface_hub==0.34.6" \
      "transformers==4.49.0" \
      "tokenizers==0.21.1" \
      "trl==0.15.2" \
      "peft==0.14.0" \
      "accelerate==1.4.0" \
      "bitsandbytes==0.45.3" \
      "datasets==4.7.0" \
      "sentencepiece==0.2.0" \
      "protobuf<6"

Post-install check:
    import torch, transformers, trl, peft, accelerate, bitsandbytes, datasets
    from trl import GRPOConfig, GRPOTrainer
    print("torch", torch.__version__)
    print("transformers", transformers.__version__)
    print("trl", trl.__version__)
    print("peft", peft.__version__)
    print("accelerate", accelerate.__version__)
    print("bitsandbytes", bitsandbytes.__version__)
    print("datasets", datasets.__version__)

Do not install or pin torch in Kaggle unless you absolutely must.
"""

import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


# -----------------------------
# Paths
# -----------------------------

def first_existing(candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")
SFT_ADAPTER_PATH = os.environ.get("SFT_ADAPTER_PATH")

if BASE_MODEL_PATH is None:
    BASE_MODEL_PATH = first_existing([
        "/kaggle/input/datasets/ebadshabbir/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2.5-3b-instruct",
        *glob.glob("/kaggle/input/*qwen*"),
        *glob.glob("/kaggle/input/**/*qwen*", recursive=True),
    ])

if SFT_ADAPTER_PATH is None:
    SFT_ADAPTER_PATH = first_existing([
        "/kaggle/input/datasets/ebadshabbir/agent-r-sft-adapter/agent_r_sft",
        "/kaggle/input/agent-r-sft-adapter/agent_r_sft",
        "/kaggle/input/agent-r-sft/agent_r_sft",
        "/kaggle/input/sft-on-gsm8k-and-math/agent_r_sft",
        "/kaggle/input/sft-on-gsm8k/agent_r_sft",
        *glob.glob("/kaggle/input/**/agent_r_sft", recursive=True),
        *glob.glob("/kaggle/input/**/*sft*", recursive=True),
    ])

if BASE_MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find Qwen base model. Set BASE_MODEL_PATH to your Kaggle input path."
    )

if SFT_ADAPTER_PATH is None:
    raise FileNotFoundError(
        "Could not find Agent-R SFT adapter. Set SFT_ADAPTER_PATH to your Kaggle input path."
    )

OUTPUT_DIR = "/kaggle/working/agent_r_grpo_fast"
FINAL_DIR = os.path.join(OUTPUT_DIR, "agent_r_v0_grpo_adapter")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Base model :", BASE_MODEL_PATH)
print("SFT adapter:", SFT_ADAPTER_PATH)
print("Output     :", FINAL_DIR)
print("Status     : paths found; continuing to load model and dataset")


# -----------------------------
# Config: keep this small/fast
# -----------------------------

MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "20"))
TRAIN_SAMPLES = int(os.environ.get("GRPO_TRAIN_SAMPLES", "200"))
SEED = 42

# If you hit OOM or the first step is too slow, reduce MAX_COMPLETION_LENGTH.
MAX_PROMPT_LENGTH = int(os.environ.get("GRPO_MAX_PROMPT_LENGTH", "256"))
MAX_COMPLETION_LENGTH = int(os.environ.get("GRPO_MAX_COMPLETION_LENGTH", "96"))


# -----------------------------
# Tokenizer + model
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    SFT_ADAPTER_PATH if os.path.exists(os.path.join(SFT_ADAPTER_PATH, "tokenizer_config.json")) else BASE_MODEL_PATH,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Status     : tokenizer loaded")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
base_model.config.use_cache = False

print("Status     : base model loaded")

base_model = prepare_model_for_kbit_training(
    base_model,
    use_gradient_checkpointing=True,
)

# Continue training the SFT LoRA adapter directly. This is simpler and faster than
# stacking/merging adapters, and it keeps the Kaggle run reliable.
model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER_PATH,
    is_trainable=True,
)
model.print_trainable_parameters()
print("Status     : SFT adapter loaded and trainable")

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


# -----------------------------
# Dataset
# -----------------------------

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoner. "
    "Solve step by step. End the final answer exactly as: #### <number>"
)


def make_prompt(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


raw = load_dataset("gsm8k", "main", split="train")
raw = raw.shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES, len(raw))))


def format_row(example):
    return {
        "prompt": make_prompt(example["question"]),
        "answer": example["answer"],
    }


train_ds = raw.map(format_row, remove_columns=raw.column_names)

print("GRPO train samples:", len(train_ds))
print("Sample prompt:")
print(train_ds[0]["prompt"][:500])


# -----------------------------
# Rewards
# -----------------------------

def completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return item.get("content", "")
    return str(completion)


def extract_number(text):
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "").strip() if numbers else None


def reward_format(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        has_marker = "####" in text
        not_too_long = len(text) < 1800
        rewards.append(0.2 if has_marker and not_too_long else 0.0)
    return rewards


def reward_correctness(completions, answer, **kwargs):
    rewards = []
    for completion, gold_answer in zip(completions, answer):
        pred = extract_number(completion_to_text(completion))
        gold = extract_number(gold_answer)
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


# -----------------------------
# GRPO
# -----------------------------

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,

    # Global batch must be divisible by num_generations. On one GPU this means
    # per_device_train_batch_size=2 with num_generations=2.
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=2,

    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    learning_rate=5e-6,
    beta=0.04,
    temperature=0.7,

    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",

    logging_steps=1,
    save_steps=10,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)


class LoudProgressCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"[GRPO] starting step {state.global_step + 1}/{args.max_steps}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        print(f"[GRPO] finished step {state.global_step}/{args.max_steps}", flush=True)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_ds,
    reward_funcs=[reward_format, reward_correctness],
    callbacks=[LoudProgressCallback()],
)

print("=" * 70)
print("Starting fast GRPO")
print(f"Steps: {MAX_STEPS}, samples: {len(train_ds)}")
print("=" * 70)

start = time.time()
trainer.train()
elapsed = time.time() - start

print("=" * 70)
print(f"GRPO finished in {elapsed / 3600:.2f} hours")
print("=" * 70)


# -----------------------------
# Save
# -----------------------------

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

meta = {
    "checkpoint_name": "agent_r_v0_grpo_adapter",
    "base_model": BASE_MODEL_PATH,
    "sft_adapter": SFT_ADAPTER_PATH,
    "method": "SFT adapter continued with fast GRPO",
    "dataset": "gsm8k/main train subset",
    "train_samples": len(train_ds),
    "max_steps": MAX_STEPS,
    "num_generations": 2,
    "max_prompt_length": MAX_PROMPT_LENGTH,
    "max_completion_length": MAX_COMPLETION_LENGTH,
    "learning_rate": 5e-6,
    "beta": 0.04,
    "runtime_hours": elapsed / 3600,
}

with open(os.path.join(FINAL_DIR, "agent_r_grpo_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

zip_path = shutil.make_archive(
    base_name="/kaggle/working/agent_r_v0_grpo_adapter",
    format="zip",
    root_dir=FINAL_DIR,
)

print("Saved adapter:", FINAL_DIR)
print("Saved zip    :", zip_path)
print(json.dumps(meta, indent=2))


# -----------------------------
# Tiny sanity check
# -----------------------------

model.eval()
questions = [
    "Janet has 3 brothers and 2 sisters. How many siblings does she have?",
    "A store sells apples for $2 each and oranges for $3 each. If you buy 4 apples and 2 oranges, how much do you spend?",
]

for question in questions:
    prompt = make_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print("-" * 70)
    print("Q:", question)
    print("A:", generated.strip())
