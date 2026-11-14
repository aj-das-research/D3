"""
Fast no-pip Kaggle runner for Agent-D.

Goal:
  Agent-D-v0 = a chat causal LM + one LoRA adapter trained in two phases:
    1. Domain-adaptive causal LM on PubMed abstracts.
    2. SFT on MedQA + PubMedQA instruction pairs.

This script intentionally avoids:
  - pip install / uninstall inside Kaggle
  - TRL / SFTTrainer version issues
  - bf16 on Tesla T4
  - device_map="auto" training hangs
  - merging LoRA into a 4-bit model between phases

Recommended Kaggle command for Mistral-7B on T4 x2 with an attached Kaggle model:
  !torchrun --nproc_per_node=2 /kaggle/working/train_agent_d.py \
      --model_path /kaggle/input/<your-mistral-7b-folder> \
      --output_dir /kaggle/working/agent_d_mistral7b_v0 \
      --profile deadline

No attached model, Internet-on fallback:
  !torchrun --nproc_per_node=2 /kaggle/working/train_agent_d.py \
      --model_path mistralai/Mistral-7B-Instruct-v0.3 \
      --output_dir /kaggle/working/agent_d_mistral7b_v0 \
      --profile deadline

Safest one-GPU command:
  !CUDA_VISIBLE_DEVICES=0 python /kaggle/working/train_agent_d.py \
      --model_path /kaggle/input/<your-mistral-7b-folder> \
      --skip_phase1 --max_medqa_rows 1500 --max_pubmedqa_rows 500

Expected output:
  /kaggle/working/agent_d_mistral7b_v0
"""

import argparse
import glob
import importlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "bitsandbytes",
]


def check_dependencies() -> None:
    missing = []
    for name in REQUIRED_MODULES:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing required Kaggle packages: "
            + ", ".join(missing)
            + ". For Kaggle, turn Internet on and run this in a setup cell, then "
            + "restart the session before launching training:\n"
            + "  !pip install -q -U bitsandbytes accelerate peft transformers datasets\n"
            + "If Internet is off, add a bitsandbytes wheel as a Kaggle dataset and "
            + "install from that local wheel instead. Mistral-7B on T4 needs "
            + "bitsandbytes because the script uses 4-bit QLoRA."
        )


check_dependencies()

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.environ.get("BASE_MODEL_PATH"))
    parser.add_argument("--output_dir", default="/kaggle/working/agent_d_v0")
    parser.add_argument("--work_dir", default="/kaggle/working/agent_d_work")
    parser.add_argument("--profile", choices=["deadline", "balanced"], default="deadline")

    parser.add_argument("--max_pubmed_docs", type=int, default=None)
    parser.add_argument("--max_medqa_rows", type=int, default=0)
    parser.add_argument("--max_pubmedqa_rows", type=int, default=0)
    parser.add_argument("--skip_phase1", action="store_true")

    parser.add_argument("--phase1_epochs", type=float, default=1.0)
    parser.add_argument("--phase2_epochs", type=float, default=2.0)
    parser.add_argument("--max_seq_len_p1", type=int, default=None)
    parser.add_argument("--max_seq_len_p2", type=int, default=None)

    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr_p1", type=float, default=2e-4)
    parser.add_argument("--lr_p2", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pubmed_json", default=os.environ.get("PUBMED_JSON"))
    parser.add_argument("--medqa_json", default=os.environ.get("MEDQA_JSON"))
    parser.add_argument("--pubmedqa_json", default=os.environ.get("PUBMEDQA_JSON"))
    # Kaggle/Colab notebooks inject kernel args such as "-f <kernel.json>".
    # Ignore unknown args so the file also works when launched from notebook magic.
    return parser.parse_known_args()[0]


ARGS = parse_args()

if ARGS.profile == "deadline":
    ARGS.max_pubmed_docs = ARGS.max_pubmed_docs or 6000
    ARGS.max_seq_len_p1 = ARGS.max_seq_len_p1 or 384
    ARGS.max_seq_len_p2 = ARGS.max_seq_len_p2 or 768
else:
    ARGS.max_pubmed_docs = ARGS.max_pubmed_docs or 30000
    ARGS.max_seq_len_p1 = ARGS.max_seq_len_p1 or 512
    ARGS.max_seq_len_p2 = ARGS.max_seq_len_p2 or 1024


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN = RANK == 0


def log(message: str = "") -> None:
    if IS_MAIN:
        print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_base_model() -> str:
    if ARGS.model_path and os.path.isdir(ARGS.model_path):
        return ARGS.model_path
    if ARGS.model_path:
        # Allow Hugging Face repo IDs such as mistralai/Mistral-7B-Instruct-v0.3
        # when Kaggle Internet is enabled and no /kaggle/input model is attached.
        return ARGS.model_path

    candidates = [
        "/kaggle/input/mistral-7b-instruct-v0-2",
        "/kaggle/input/mistral-7b-instruct-v0-3",
        "/kaggle/input/mistral-7b-instruct",
        *glob.glob("/kaggle/input/*mistral*7b*", recursive=False),
        *glob.glob("/kaggle/input/**/*mistral*7b*", recursive=True),
        "/kaggle/input/datasets/ebadshabbir/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2.5-3b-instruct",
        *glob.glob("/kaggle/input/*qwen*", recursive=False),
        *glob.glob("/kaggle/input/**/*qwen*", recursive=True),
    ]
    for candidate in candidates:
        path = Path(candidate)
        if (path / "config.json").exists():
            return str(path)

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for config_path in input_root.rglob("config.json"):
            text = str(config_path.parent).lower()
            if ("mistral" in text and "7b" in text) or ("qwen" in text and "3b" in text):
                return str(config_path.parent)

    return os.environ.get("DEFAULT_HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")


find_qwen_model = find_base_model


def select_limit(dataset, limit: int, seed: int):
    if limit and limit > 0 and len(dataset) > limit:
        return dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def normalize_options(options):
    if isinstance(options, dict):
        return [(str(k), str(v)) for k, v in sorted(options.items())]
    if isinstance(options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [(letters[i], str(v)) for i, v in enumerate(options)]
    return [("A", str(options))]


def load_pubmed_dataset(tokenizer):
    log("\nLoading Phase 1 PubMed abstracts...")
    if ARGS.pubmed_json and os.path.exists(ARGS.pubmed_json):
        raw = load_dataset("json", data_files=ARGS.pubmed_json, split="train")
    else:
        raw = load_dataset("ccdv/pubmed-summarization", "document", split="train")

    raw = select_limit(raw, ARGS.max_pubmed_docs, ARGS.seed)
    text_column = None
    for candidate in ["article", "abstract", "text", "content"]:
        if candidate in raw.column_names:
            text_column = candidate
            break
    if text_column is None:
        raise ValueError(f"Could not find PubMed text column. Columns: {raw.column_names}")

    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=ARGS.max_seq_len_p1,
            padding=False,
        )

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw.column_names,
        load_from_cache_file=False,
        desc="Tokenizing PubMed",
    )
    log(f"Phase 1 examples: {len(tokenized):,}")
    return tokenized


SYSTEM_D = (
    "You are a specialist in biomedical literature and clinical medicine. "
    "Answer precisely from medical evidence and clinical principles."
)


def chat_text(tokenizer, user: str, assistant: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_D},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_sft_dataset(tokenizer):
    log("\nLoading Phase 2 SFT datasets...")

    if ARGS.medqa_json and os.path.exists(ARGS.medqa_json):
        medqa = load_dataset("json", data_files=ARGS.medqa_json, split="train")
    else:
        medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    medqa = select_limit(medqa, ARGS.max_medqa_rows, ARGS.seed)

    def format_medqa(example):
        options = normalize_options(example.get("options", {}))
        options_text = "\n".join(f"{letter}. {value}" for letter, value in options)
        option_map = {letter: value for letter, value in options}
        answer_letter = str(example.get("answer_idx", "")).strip()
        answer_text = str(example.get("answer", "")).strip()
        if answer_letter in option_map:
            answer_text = option_map[answer_letter]
        elif not answer_text and options:
            answer_letter, answer_text = options[0]

        user = (
            f"{example.get('question', '')}\n\n"
            f"Options:\n{options_text}\n\n"
            "Choose the best answer."
        )
        assistant = f"The correct answer is {answer_letter}. {answer_text}".strip()
        return {"text": chat_text(tokenizer, user, assistant)}

    medqa = medqa.map(
        format_medqa,
        remove_columns=medqa.column_names,
        load_from_cache_file=False,
        desc="Formatting MedQA",
    )

    if ARGS.pubmedqa_json and os.path.exists(ARGS.pubmedqa_json):
        pubmedqa = load_dataset("json", data_files=ARGS.pubmedqa_json, split="train")
    else:
        pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    pubmedqa = select_limit(pubmedqa, ARGS.max_pubmedqa_rows, ARGS.seed)

    def format_pubmedqa(example):
        context = example.get("context", {})
        if isinstance(context, dict):
            contexts = context.get("contexts", [])
            context_text = " ".join(str(x) for x in contexts)
        elif isinstance(context, list):
            context_text = " ".join(str(x) for x in context)
        else:
            context_text = str(context)
        context_text = context_text[:900]

        question = str(example.get("question", ""))
        decision = str(example.get("final_decision", "")).strip().lower()
        long_answer = str(example.get("long_answer", "")).strip()[:650]
        user = (
            "Based on the research context, answer yes, no, or maybe.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}"
        )
        assistant = f"{decision.capitalize()}. {long_answer}".strip()
        return {"text": chat_text(tokenizer, user, assistant)}

    pubmedqa = pubmedqa.map(
        format_pubmedqa,
        remove_columns=pubmedqa.column_names,
        load_from_cache_file=False,
        desc="Formatting PubMedQA",
    )

    full = concatenate_datasets([medqa, pubmedqa]).shuffle(seed=ARGS.seed)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=ARGS.max_seq_len_p2,
            padding=False,
        )

    tokenized = full.map(
        tokenize,
        batched=True,
        remove_columns=full.column_names,
        load_from_cache_file=False,
        desc="Tokenizing SFT",
    )
    log(f"Phase 2 examples: {len(tokenized):,}")
    return tokenized


class ProgressCallback(TrainerCallback):
    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        if IS_MAIN:
            log(f"\n{self.phase_name} started. max_steps={state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not IS_MAIN or logs is None or self.start is None:
            return
        elapsed = time.time() - self.start
        step = max(state.global_step, 1)
        total = max(state.max_steps, step)
        eta = elapsed / step * max(total - step, 0)
        loss = logs.get("loss")
        loss_text = f"{loss:.4f}" if isinstance(loss, float) else "n/a"
        log(
            f"{self.phase_name}: step {state.global_step}/{total} "
            f"loss={loss_text} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )


def make_training_args(output_dir: str, epochs: float, lr: float, fp16: bool, bf16: bool):
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": ARGS.per_device_batch,
        "gradient_accumulation_steps": ARGS.grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "optim": "paged_adamw_8bit",
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": True,
        "logging_steps": ARGS.logging_steps,
        "save_steps": ARGS.save_steps,
        "save_total_limit": 1,
        "report_to": "none",
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "group_by_length": True,
        "disable_tqdm": False,
        "ddp_find_unused_parameters": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return TrainingArguments(**filtered)


def make_trainer(model, training_args, train_dataset, tokenizer, phase_name: str):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "callbacks": [ProgressCallback(phase_name)],
    }
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    set_seed(ARGS.seed)
    model_path = find_qwen_model()
    work_dir = Path(ARGS.work_dir)
    output_dir = Path(ARGS.output_dir)
    phase1_dir = work_dir / "phase1_pubmed"
    phase2_dir = work_dir / "phase2_sft"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
        cap = torch.cuda.get_device_capability(LOCAL_RANK)
        gpu_name = torch.cuda.get_device_name(LOCAL_RANK)
        bf16_ok = cap[0] >= 8
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        gpu_name = "CPU"
        bf16_ok = False
        compute_dtype = torch.float32

    fp16 = compute_dtype == torch.float16
    bf16 = compute_dtype == torch.bfloat16

    log("=" * 72)
    log("Agent-D fast no-pip training")
    log("=" * 72)
    log(f"Model path      : {model_path}")
    log(f"Output dir      : {output_dir}")
    log(f"Profile         : {ARGS.profile}")
    log(f"World size      : {WORLD_SIZE}")
    log(f"GPU             : {gpu_name}")
    log(f"Precision       : {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}")
    log(f"PubMed docs     : {0 if ARGS.skip_phase1 else ARGS.max_pubmed_docs}")
    log(f"SFT epochs      : {ARGS.phase2_epochs}")
    log(f"Seq lens        : p1={ARGS.max_seq_len_p1}, p2={ARGS.max_seq_len_p2}")
    log("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    device_map = {"": LOCAL_RANK} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if IS_MAIN:
        model.print_trainable_parameters()

    if not ARGS.skip_phase1:
        phase1_dataset = load_pubmed_dataset(tokenizer)
        args1 = make_training_args(
            str(phase1_dir),
            ARGS.phase1_epochs,
            ARGS.lr_p1,
            fp16=fp16,
            bf16=bf16,
        )
        trainer1 = make_trainer(model, args1, phase1_dataset, tokenizer, "Phase 1 PubMed DAPT")
        t0 = time.time()
        trainer1.train()
        log(f"Phase 1 finished in {(time.time() - t0) / 60:.1f} minutes")
        trainer1.save_model(str(phase1_dir))
        if IS_MAIN:
            tokenizer.save_pretrained(str(phase1_dir))
        barrier()

    phase2_dataset = load_sft_dataset(tokenizer)
    args2 = make_training_args(
        str(phase2_dir),
        ARGS.phase2_epochs,
        ARGS.lr_p2,
        fp16=fp16,
        bf16=bf16,
    )
    trainer2 = make_trainer(model, args2, phase2_dataset, tokenizer, "Phase 2 MedQA+PubMedQA SFT")
    t0 = time.time()
    trainer2.train()
    log(f"Phase 2 finished in {(time.time() - t0) / 60:.1f} minutes")

    trainer2.save_model(str(output_dir))
    if IS_MAIN:
        tokenizer.save_pretrained(str(output_dir))
    barrier()

    if IS_MAIN:
        meta = {
            "checkpoint_name": "Agent-D-v0",
            "base_model": model_path,
            "profile": ARGS.profile,
            "training": {
                "phase1": {
                    "skipped": ARGS.skip_phase1,
                    "dataset": "ccdv/pubmed-summarization/document"
                    if not ARGS.pubmed_json
                    else ARGS.pubmed_json,
                    "max_docs": 0 if ARGS.skip_phase1 else ARGS.max_pubmed_docs,
                    "epochs": ARGS.phase1_epochs,
                    "max_seq_len": ARGS.max_seq_len_p1,
                    "learning_rate": ARGS.lr_p1,
                },
                "phase2": {
                    "datasets": [
                        "GBaker/MedQA-USMLE-4-options"
                        if not ARGS.medqa_json
                        else ARGS.medqa_json,
                        "qiaojin/PubMedQA/pqa_labeled"
                        if not ARGS.pubmedqa_json
                        else ARGS.pubmedqa_json,
                    ],
                    "epochs": ARGS.phase2_epochs,
                    "max_seq_len": ARGS.max_seq_len_p2,
                    "learning_rate": ARGS.lr_p2,
                },
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": sorted(list(lora.target_modules)),
                },
                "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
                "world_size": WORLD_SIZE,
            },
            "agent_role": "Domain specialist: biomedical literature and clinical QA",
            "expected_failure_mode": (
                "Strong medical pattern matching; weaker multi-step cross-domain reasoning."
            ),
        }
        with open(output_dir / "agent_d_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log("\nAgent-D-v0 saved.")
        log(f"Final adapter: {output_dir}")
        log("Download this Kaggle output folder before the session ends.")


if __name__ == "__main__":
    main()
