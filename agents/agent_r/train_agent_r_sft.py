"""
Agent-R supervised fine-tuning: the stage before GRPO refinement
(paper Section 3.2; Appendix D.1).

Trains a LoRA adapter on GSM8K worked solutions on top of
Qwen2.5-3B-Instruct. The resulting adapter is the input that
train_agent_r_grpo.py expects as SFT_ADAPTER_PATH.

Run with:
    pip install transformers==4.47.0 trl==0.12.0 peft==0.14.0 datasets accelerate sentencepiece
    BASE_MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
    OUTPUT_DIR=outputs/agent_r_sft \
    python agents/agent_r/train_agent_r_sft.py

Settings match the run that produced the released SFT adapter: LoRA r=16,
alpha=32, dropout 0.05 on the attention projections; 2 epochs; learning rate
2e-4; effective batch size 16; bf16.
"""

import os

os.environ.setdefault("WANDB_DISABLED", "true")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/agent_r_sft")


def format_example(example):
    return {"text": f"Question: {example['question']}\nSolution: {example['answer']}"}


def main() -> None:
    dataset = load_dataset("gsm8k", "main", split="train").map(format_example)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        warmup_ratio=0.05,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved Agent-R SFT adapter to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
