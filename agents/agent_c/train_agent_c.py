"""
Agent-C training: calibration specialist (paper Section 3.2, Eq. 1; Appendix D.1).

Runs on Kaggle without extra pip installs.

Goal:
  Agent-C-v0 = Qwen2.5-3B-Instruct + one LoRA adapter trained as a
  calibrated multiple-choice scorer.

Training objective:
  loss = cross_entropy(choice_logits, gold_letter) + ece_weight * ECE_loss

Post-training:
  Fit one scalar temperature on held-out validation choice logits, then save
  it beside the adapter. Inference should use softmax(choice_logits / T), so
  Agent-C returns both an answer letter and an honest confidence.

Recommended Kaggle command:
  !CUDA_VISIBLE_DEVICES=0 python /kaggle/working/train_agent_c.py \
      --mmlu_json /kaggle/input/YOUR_DATASET/mmlu_pro_hard_subset.json \
      --split_manifest /kaggle/input/YOUR_SPLIT_DATASET/mmlu_pro_hard_split_manifest.csv

Expected output:
  /kaggle/working/agent_c_v0
  /kaggle/working/agent_c_v0/temperature.json
"""

import argparse
import ast
import csv
import glob
import importlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for helper_dir in (
    Path(__file__).resolve().parent,
    PROJECT_ROOT / "ddd",
    Path.cwd() / "ddd",
):
    sys.path.insert(0, str(helper_dir))

from split_guard import assert_disjoint_question_ids, select_records


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
            + ". Use a Kaggle image/session that already has these packages, "
            + "or install once and restart the kernel before running this file."
        )


check_dependencies()

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


LETTERS = "ABCDEFGHIJ"
SYSTEM_C = (
    "You are Agent-C, a cautious calibrated analyst. "
    "Choose the best answer, but keep your uncertainty honest."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.environ.get("BASE_MODEL_PATH"))
    parser.add_argument("--output_dir", default="/kaggle/working/agent_c_v0")
    parser.add_argument("--work_dir", default="/kaggle/working/agent_c_work")
    parser.add_argument("--profile", choices=["deadline", "balanced"], default="deadline")

    parser.add_argument("--mmlu_json", default=os.environ.get("MMLU_JSON"))
    parser.add_argument("--mmlu_csv", default=os.environ.get("MMLU_CSV"))
    parser.add_argument("--hf_split", default="test")
    parser.add_argument(
        "--split_manifest",
        default=os.environ.get("SPLIT_MANIFEST"),
        help="Immutable question_id/split CSV used instead of a random split.",
    )
    parser.add_argument(
        "--split_manifest_sha256",
        default=os.environ.get("SPLIT_MANIFEST_SHA256"),
    )
    parser.add_argument(
        "--allow_legacy_random_split",
        action="store_true",
        help="Explicit escape hatch for runs without a manifest; not valid for reported results.",
    )

    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ece_weight", type=float, default=0.15)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--temperature_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Kaggle/Colab notebooks inject kernel args such as "-f <kernel.json>".
    return parser.parse_known_args()[0]


ARGS = parse_args()

if ARGS.profile == "deadline":
    ARGS.max_rows = ARGS.max_rows or 2500
    ARGS.max_seq_len = ARGS.max_seq_len or 768
    ARGS.num_train_epochs = ARGS.num_train_epochs or 1.0
else:
    ARGS.max_rows = ARGS.max_rows or 6000
    ARGS.max_seq_len = ARGS.max_seq_len or 1024
    ARGS.num_train_epochs = ARGS.num_train_epochs or 2.0


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


def find_qwen_model() -> str:
    if ARGS.model_path and os.path.isdir(ARGS.model_path):
        return ARGS.model_path

    candidates = [
        "/kaggle/input/datasets/ebadshabbir/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2-5-3b-instruct",
        "/kaggle/input/qwen2.5-3b-instruct",
        "/kaggle/input/qwen25-3b-instruct",
        *glob.glob("/kaggle/input/*qwen*", recursive=False),
        *glob.glob("/kaggle/input/**/*qwen*", recursive=True),
        str(Path.cwd() / "qwen-model"),
        str(Path.cwd() / "Qwen2.5-3B-Instruct"),
    ]
    for candidate in candidates:
        path = Path(candidate)
        if (path / "config.json").exists():
            return str(path)

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for config_path in input_root.rglob("config.json"):
            text = str(config_path.parent).lower()
            if "qwen" in text and "3b" in text:
                return str(config_path.parent)

    raise FileNotFoundError(
        "Could not find Qwen under /kaggle/input. Set BASE_MODEL_PATH or pass "
        "--model_path /kaggle/input/.../qwen2-5-3b-instruct"
    )


def find_mmlu_file() -> Optional[str]:
    explicit = [ARGS.mmlu_json, ARGS.mmlu_csv]
    for path in explicit:
        if path and os.path.exists(path):
            return path

    candidates = [
        *glob.glob("/kaggle/input/**/mmlu_pro_hard_subset.json", recursive=True),
        *glob.glob("/kaggle/input/**/mmlu_pro_hard_subset.csv", recursive=True),
        str(Path.cwd() / "data" / "mmlu_pro_hard_subset.json"),
        str(Path.cwd() / "data" / "mmlu_pro_hard_subset.csv"),
        str(Path.cwd().parent / "data" / "mmlu_pro_hard_subset.json"),
        str(Path.cwd().parent / "data" / "mmlu_pro_hard_subset.csv"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_split_manifest() -> Optional[str]:
    candidates = [
        ARGS.split_manifest,
        str(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "mmlu_pro_hard_split_manifest.csv"
        ),
        str(Path.cwd() / "data" / "splits" / "mmlu_pro_hard_split_manifest.csv"),
        str(Path.cwd() / "mmlu_pro_hard_split_manifest.csv"),
        *glob.glob(
            "/kaggle/input/**/mmlu_pro_hard_split_manifest.csv",
            recursive=True,
        ),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def parse_options(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, dict):
        return [str(value[k]) for k in sorted(value)]
    text = str(value)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
            if isinstance(parsed, dict):
                return [str(parsed[k]) for k in sorted(parsed)]
        except Exception:
            pass
    return [text]


def infer_answer_letter(row: Dict, options: List[str]) -> Optional[str]:
    raw_answer = str(row.get("answer", "")).strip()
    if raw_answer.upper() in LETTERS:
        idx = LETTERS.index(raw_answer.upper())
        if idx < len(options):
            return raw_answer.upper()

    raw_idx = row.get("answer_index", row.get("answer_idx", None))
    if raw_idx is not None and str(raw_idx).strip() != "":
        try:
            idx = int(raw_idx)
            if 0 <= idx < len(options):
                return LETTERS[idx]
        except Exception:
            pass

    normalized = raw_answer.lower()
    for idx, option in enumerate(options):
        if normalized and normalized == str(option).strip().lower():
            return LETTERS[idx]
    return None


def normalize_record(row: Dict) -> Optional[Dict]:
    options = parse_options(row.get("options", []))
    if len(options) < 2:
        return None
    options = options[: len(LETTERS)]

    answer = infer_answer_letter(row, options)
    if answer is None:
        return None

    question = str(row.get("question", "")).strip()
    if not question:
        return None

    return {
        "question_id": str(row.get("question_id", row.get("id", ""))),
        "question": question,
        "options": options,
        "answer": answer,
        "category": str(row.get("category", "")),
        "difficulty_score": row.get("difficulty_score", row.get("final_difficulty", None)),
    }


def load_records_from_file(path: str) -> List[Dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    elif suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    elif suffix == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    else:
        raise ValueError(f"Unsupported MMLU file type: {path}")

    normalized = [normalize_record(dict(row)) for row in rows]
    return [row for row in normalized if row is not None]


def load_mmlu_records() -> Tuple[List[Dict], str]:
    path = find_mmlu_file()
    if path:
        records = load_records_from_file(path)
        source = path
    else:
        log("No local MMLU hard-subset file found. Falling back to TIGER-Lab/MMLU-Pro.")
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=ARGS.hf_split)
        records = []
        for row in ds:
            normalized = normalize_record(dict(row))
            if normalized is not None:
                records.append(normalized)
        source = f"TIGER-Lab/MMLU-Pro/{ARGS.hf_split}"

    if not records:
        raise ValueError("No usable MMLU-Pro records found.")

    return records, source


def split_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
    if len(records) < 20:
        raise ValueError("Need at least 20 records so Agent-C can fit a validation temperature.")

    manifest_path = find_split_manifest()
    if manifest_path:
        train_records, train_report = select_records(
            records,
            manifest_path,
            {"train"},
            expected_sha256=ARGS.split_manifest_sha256,
        )
        val_records, validation_report = select_records(
            records,
            manifest_path,
            {"validation"},
            expected_sha256=ARGS.split_manifest_sha256,
        )
        assert_disjoint_question_ids(
            {"agent_c_train": train_records, "agent_c_validation": val_records}
        )
        random.Random(ARGS.seed).shuffle(train_records)
        if ARGS.max_rows and len(train_records) > ARGS.max_rows:
            train_records = train_records[: ARGS.max_rows]
        split_report = {
            "mode": "strict_manifest",
            "manifest": train_report["manifest"],
            "manifest_sha256": train_report["manifest_sha256"],
            "train_rows": len(train_records),
            "validation_rows": len(val_records),
            "train_selection": train_report,
            "validation_selection": validation_report,
        }
        return train_records, val_records, split_report

    if not ARGS.allow_legacy_random_split:
        raise FileNotFoundError(
            "No split manifest found. Pass --split_manifest; use "
            "--allow_legacy_random_split only for exploratory runs."
        )

    shuffled = list(records)
    random.Random(ARGS.seed).shuffle(shuffled)
    if ARGS.max_rows and len(shuffled) > ARGS.max_rows:
        shuffled = shuffled[: ARGS.max_rows]
    n_val = max(32, int(round(len(shuffled) * ARGS.val_fraction)))
    n_val = min(n_val, max(1, len(shuffled) // 3))
    return (
        shuffled[n_val:],
        shuffled[:n_val],
        {
            "mode": "legacy_random_split_not_for_reported_results",
            "seed": ARGS.seed,
            "train_rows": len(shuffled) - n_val,
            "validation_rows": n_val,
        },
    )


def format_prompt(tokenizer, record: Dict) -> str:
    options_text = "\n".join(
        f"{LETTERS[i]}. {option}" for i, option in enumerate(record["options"])
    )
    user = (
        "Answer this multiple-choice question.\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        "Respond with only the single letter of the best answer."
    )
    messages = [
        {"role": "system", "content": SYSTEM_C},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_choice_token_ids(tokenizer, n_choices: int) -> List[int]:
    token_ids = []
    for letter in LETTERS[:n_choices]:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Choice label {letter!r} did not tokenize as one token: {ids}. "
                "This script expects Qwen-style single-token A-J labels."
            )
        token_ids.append(ids[0])
    return token_ids


class ChoicePromptDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_seq_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        prompt = format_prompt(self.tokenizer, record)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": LETTERS.index(record["answer"]),
        }


class ChoiceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def choice_logits_from_outputs(outputs, attention_mask, choice_token_ids: torch.Tensor) -> torch.Tensor:
    logits = outputs.logits
    last_positions = attention_mask.long().sum(dim=1) - 1
    row_idx = torch.arange(logits.shape[0], device=logits.device)
    next_token_logits = logits[row_idx, last_positions]
    return next_token_logits.index_select(dim=-1, index=choice_token_ids.to(logits.device))


def ece_training_loss(choice_logits: torch.Tensor, labels: torch.Tensor, n_bins: int) -> torch.Tensor:
    probs = F.softmax(choice_logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == labels).float()

    loss = choice_logits.new_tensor(0.0)
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == 0:
            mask = (confidences.detach() >= lower) & (confidences.detach() <= upper)
        else:
            mask = (confidences.detach() > lower) & (confidences.detach() <= upper)
        if mask.any():
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            weight = mask.float().mean()
            loss = loss + weight * (avg_conf - avg_acc).pow(2)
    return loss


class CalibrationTrainer(Trainer):
    def __init__(
        self,
        *args,
        choice_token_ids: List[int],
        ece_weight: float,
        ece_bins: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.choice_token_ids = torch.tensor(choice_token_ids, dtype=torch.long)
        self.ece_weight = ece_weight
        self.ece_bins = ece_bins

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        choice_logits = choice_logits_from_outputs(
            outputs,
            inputs["attention_mask"],
            self.choice_token_ids,
        )
        ce_loss = F.cross_entropy(choice_logits, labels)
        cal_loss = ece_training_loss(choice_logits, labels, self.ece_bins)
        loss = ce_loss + self.ece_weight * cal_loss

        if return_outputs:
            return loss, {
                "choice_logits": choice_logits.detach(),
                "ce_loss": ce_loss.detach(),
                "ece_loss": cal_loss.detach(),
            }
        return loss


class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()
        log(f"\nAgent-C training started. max_steps={state.max_steps}")

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
            f"Agent-C: step {state.global_step}/{total} "
            f"loss={loss_text} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )


def make_training_args(output_dir: str, fp16: bool, bf16: bool):
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": ARGS.num_train_epochs,
        "per_device_train_batch_size": ARGS.per_device_batch,
        "gradient_accumulation_steps": ARGS.grad_accum,
        "learning_rate": ARGS.learning_rate,
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


def make_trainer(model, training_args, train_dataset, tokenizer, choice_token_ids):
    collator = ChoiceCollator(tokenizer)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
        "callbacks": [ProgressCallback()],
        "choice_token_ids": choice_token_ids,
        "ece_weight": ARGS.ece_weight,
        "ece_bins": ARGS.ece_bins,
    }
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return CalibrationTrainer(**kwargs)


@torch.no_grad()
def collect_choice_logits(
    model,
    dataset: Dataset,
    tokenizer,
    choice_token_ids: List[int],
    batch_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    collator = ChoiceCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    token_ids = torch.tensor(choice_token_ids, dtype=torch.long)

    all_logits = []
    all_labels = []
    device = next(model.parameters()).device
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = choice_logits_from_outputs(outputs, batch["attention_mask"], token_ids)
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    n_bins: int,
) -> Dict[str, float]:
    scaled = logits.float() / float(temperature)
    probs = F.softmax(scaled, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == labels).float()
    one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).float()

    ece = torch.tensor(0.0)
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            avg_conf = confidences[mask].mean()
            avg_acc = correct[mask].mean()
            weight = mask.float().mean()
            ece = ece + weight * (avg_conf - avg_acc).abs()

    return {
        "accuracy": float(correct.mean().item()),
        "nll": float(F.cross_entropy(scaled, labels).item()),
        "ece": float(ece.item()),
        "avg_confidence": float(confidences.mean().item()),
        "brier": float(((probs - one_hot) ** 2).sum(dim=-1).mean().item()),
    }


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = logits.float().to(device)
    labels = labels.long().to(device)

    log_temp = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_temp], lr=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        optimizer.step()

    return float(log_temp.exp().clamp(0.05, 10.0).detach().cpu().item())


def score_record(
    model,
    tokenizer,
    record: Dict,
    choice_token_ids: List[int],
    temperature: float,
) -> Dict:
    dataset = ChoicePromptDataset([record], tokenizer, ARGS.max_seq_len)
    logits, _ = collect_choice_logits(
        model,
        dataset,
        tokenizer,
        choice_token_ids,
        batch_size=1,
    )
    probs = F.softmax(logits[0] / temperature, dim=-1)
    confidence, pred_idx = probs.max(dim=-1)
    return {
        "answer": LETTERS[int(pred_idx.item())],
        "confidence": float(confidence.item()),
        "gold": record["answer"],
    }


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    set_seed(ARGS.seed)
    model_path = find_qwen_model()
    work_dir = Path(ARGS.work_dir)
    output_dir = Path(ARGS.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, data_source = load_mmlu_records()
    train_records, val_records, split_report = split_records(records)

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
    log("Agent-C calibration training")
    log("=" * 72)
    log(f"Model path       : {model_path}")
    log(f"Data source      : {data_source}")
    log(f"Split mode       : {split_report['mode']}")
    if split_report.get("manifest_sha256"):
        log(f"Split manifest   : {split_report['manifest_sha256']}")
    log(f"Output dir       : {output_dir}")
    log(f"Profile          : {ARGS.profile}")
    log(f"World size       : {WORLD_SIZE}")
    log(f"GPU              : {gpu_name}")
    log(f"Precision        : {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}")
    log(f"Train / val rows : {len(train_records):,} / {len(val_records):,}")
    log(f"Seq len          : {ARGS.max_seq_len}")
    log(f"Loss             : CE + {ARGS.ece_weight} * ECE({ARGS.ece_bins} bins)")
    log("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    max_choices = min(max(len(r["options"]) for r in records), len(LETTERS))
    choice_token_ids = get_choice_token_ids(tokenizer, max_choices)
    log(f"Choice labels    : {LETTERS[:max_choices]}")
    log(f"Choice token ids : {choice_token_ids}")

    train_dataset = ChoicePromptDataset(train_records, tokenizer, ARGS.max_seq_len)
    val_dataset = ChoicePromptDataset(val_records, tokenizer, ARGS.max_seq_len)

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

    training_args = make_training_args(str(work_dir / "trainer"), fp16=fp16, bf16=bf16)
    trainer = make_trainer(model, training_args, train_dataset, tokenizer, choice_token_ids)

    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    log(f"\nAgent-C training finished in {train_minutes:.1f} minutes")

    trainer.save_model(str(output_dir))
    if IS_MAIN:
        tokenizer.save_pretrained(str(output_dir))
    barrier()

    if IS_MAIN:
        log("\nCollecting validation logits for temperature scaling...")
        val_logits, val_labels = collect_choice_logits(
            model,
            val_dataset,
            tokenizer,
            choice_token_ids,
            batch_size=max(1, min(8, ARGS.per_device_batch * 4)),
        )

        before = calibration_metrics(
            val_logits,
            val_labels,
            temperature=1.0,
            n_bins=ARGS.ece_bins,
        )
        temperature = fit_temperature(
            val_logits,
            val_labels,
            steps=ARGS.temperature_steps,
        )
        after = calibration_metrics(
            val_logits,
            val_labels,
            temperature=temperature,
            n_bins=ARGS.ece_bins,
        )

        temperature_payload = {
            "temperature": temperature,
            "choice_labels": LETTERS[:max_choices],
            "choice_token_ids": choice_token_ids,
            "metrics_before_temperature": before,
            "metrics_after_temperature": after,
            "validation_rows": len(val_records),
        }
        with open(output_dir / "temperature.json", "w", encoding="utf-8") as f:
            json.dump(temperature_payload, f, indent=2)

        meta = {
            "checkpoint_name": "Agent-C-v0",
            "base_model": model_path,
            "data_source": data_source,
            "split_protocol": split_report,
            "profile": ARGS.profile,
            "training": {
                "train_rows": len(train_records),
                "val_rows": len(val_records),
                "epochs": ARGS.num_train_epochs,
                "max_seq_len": ARGS.max_seq_len,
                "learning_rate": ARGS.learning_rate,
                "loss": {
                    "cross_entropy_weight": 1.0,
                    "ece_weight": ARGS.ece_weight,
                    "ece_bins": ARGS.ece_bins,
                },
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": sorted(list(lora.target_modules)),
                },
                "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
                "world_size": WORLD_SIZE,
                "runtime_minutes": train_minutes,
            },
            "temperature": temperature_payload,
            "agent_role": "Calibration specialist: calibrated answer confidence for multiple-choice QA",
            "inference_rule": "Compute A-J next-token logits, divide by temperature, softmax, then return argmax letter and max probability.",
        }
        with open(output_dir / "agent_c_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log("\nValidation metrics before temperature:")
        log(json.dumps(before, indent=2))
        log("\nValidation metrics after temperature:")
        log(json.dumps(after, indent=2))

        log("\nTiny sanity check:")
        for record in val_records[:3]:
            scored = score_record(model, tokenizer, record, choice_token_ids, temperature)
            qid = record["question_id"] or "(no id)"
            log(
                f"  {qid}: pred={scored['answer']} gold={scored['gold']} "
                f"conf={scored['confidence']:.3f}"
            )

        log("\nAgent-C-v0 saved.")
        log(f"Final adapter      : {output_dir}")
        log(f"Temperature config : {output_dir / 'temperature.json'}")


if __name__ == "__main__":
    main()
