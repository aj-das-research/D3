"""
Hugging Face / PEFT backend for the co-evolution loop in ddd/coevolution.py.

Each agent is a trainable LoRA adapter on its own base model. Agents that share
a base model (same-family setting) share one loaded copy of it; the
different-family setting loads a second base for Agent-D. Letters are scored
from next-token logits over the option label tokens, as in the static runs.

Update rule for one agent (paper Eq. 9):
    loss = CE(choice_logits, gold)                       (all agents)
         + lambda_ece * binned ECE loss                  (Agent-C only, Eq. 1)
         + lambda_div * mean_q max(0, tau_JS - Jbar(q))  (diversity hinge)
where Jbar(q) is the mean pairwise normalised JS between the updated agent's
current distribution and its peers' distributions from the start of the
iteration (held fixed). Only that agent's LoRA parameters receive gradients.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "static_disagreement"))
sys.path.insert(0, str(REPO_ROOT / "ddd"))

from diversity_regularizer import torch_js_normalized  # noqa: E402
from run_heterogeneous import build_prompt, letter_token_ids, load_temperature  # noqa: E402

EPS = 1e-12


def ece_training_loss(choice_logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Binned calibration loss, identical to agents/agent_c/train_agent_c.py."""
    probs = F.softmax(choice_logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == labels).float()
    loss = choice_logits.new_tensor(0.0)
    for i in range(n_bins):
        lower, upper = i / n_bins, (i + 1) / n_bins
        conf = confidences.detach()
        mask = (conf >= lower) & (conf <= upper) if i == 0 else (conf > lower) & (conf <= upper)
        if mask.any():
            loss = loss + mask.float().mean() * (confidences[mask].mean() - correct[mask].mean()).pow(2)
    return loss


class HFAgentBackend:
    def __init__(
        self,
        agent_specs: Dict[str, Dict],
        learning_rate: float = 1e-5,
        micro_batch_size: int = 4,
        lambda_ece: float = 0.15,
        load_in_4bit: bool = True,
        max_length: int = 1024,
        seed: int = 42,
    ):
        """
        agent_specs: {"R": {"base": path, "adapter": path}, "D": {...}, "C": {...}}
        Agent order is the dict order.
        """
        torch.manual_seed(seed)
        self.agents = tuple(agent_specs)
        self.specs = agent_specs
        self.lr = learning_rate
        self.micro = micro_batch_size
        self.lambda_ece = lambda_ece
        self.max_length = max_length
        self.models: Dict[str, PeftModel] = {}
        self.tokenizers: Dict[str, object] = {}
        self.model_of: Dict[str, str] = {}
        self.temperature: Dict[str, float] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

        quant = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            if load_in_4bit else None
        )
        for agent, spec in agent_specs.items():
            base = str(spec["base"])
            adapter = Path(spec["adapter"])
            name = self._adapter_name(agent)
            if base not in self.models:
                tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
                tok.padding_side = "left"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    base, quantization_config=quant, device_map="auto", trust_remote_code=True,
                    torch_dtype=torch.float16 if load_in_4bit else torch.bfloat16,
                )
                if load_in_4bit:
                    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
                model = PeftModel.from_pretrained(model, str(adapter), adapter_name=name, is_trainable=True)
                self.models[base] = model
                self.tokenizers[base] = tok
            else:
                self.models[base].load_adapter(str(adapter), adapter_name=name, is_trainable=True)
            self.model_of[agent] = base
            self.temperature[agent] = load_temperature(adapter)

    @staticmethod
    def _adapter_name(agent: str) -> str:
        return f"agent_{agent}"

    def _activate(self, agent: str, train: bool):
        model = self.models[self.model_of[agent]]
        name = self._adapter_name(agent)
        model.set_adapter(name)
        for pname, param in model.named_parameters():
            param.requires_grad = train and "lora_" in pname and f".{name}." in pname
        model.train(train)
        return model, self.tokenizers[self.model_of[agent]]

    def _choice_logits(self, model, tok, agent: str, questions: List[Dict]) -> torch.Tensor:
        prompts = [build_prompt(tok, agent, q["question"], q["options"]) for q in questions]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=self.max_length).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        k = max(len(q["options"]) for q in questions)
        ids = letter_token_ids(tok, k)
        choice = logits[:, ids].float()
        mask = torch.full_like(choice, float("-inf"))
        for i, q in enumerate(questions):
            mask[i, : len(q["options"])] = 0.0
        return choice + mask

    @torch.no_grad()
    def predict(self, agent: str, questions: List[Dict]) -> List[np.ndarray]:
        model, tok = self._activate(agent, train=False)
        out: List[np.ndarray] = []
        for start in range(0, len(questions), 8):
            chunk = questions[start : start + 8]
            logits = self._choice_logits(model, tok, agent, chunk)
            probs = F.softmax(logits / max(self.temperature[agent], 1e-6), dim=-1).cpu().numpy()
            out.extend(p[: len(q["options"])] for p, q in zip(probs, chunk))
        return out

    def update(self, agent, questions, peer_probs, steps, lambda_div, tau_js) -> Dict:
        model, tok = self._activate(agent, train=True)
        if agent not in self.optimizers:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers[agent] = torch.optim.AdamW(params, lr=self.lr)
        opt = self.optimizers[agent]
        letters = "ABCDEFGHIJ"
        logs = {"loss": [], "task": [], "div": []}
        order = np.arange(len(questions))
        for step in range(steps):
            start = (step * self.micro) % len(order)
            idx = order[start : start + self.micro]
            if len(idx) < self.micro:
                idx = np.concatenate([idx, order[: self.micro - len(idx)]])
            batch = [questions[i] for i in idx]
            logits = self._choice_logits(model, tok, agent, batch)
            labels = torch.tensor([letters.index(q["answer"]) for q in batch], device=logits.device)
            task = F.cross_entropy(logits, labels)
            if agent == "C" and self.lambda_ece > 0:
                task = task + self.lambda_ece * ece_training_loss(logits, labels)
            div = logits.new_tensor(0.0)
            if lambda_div > 0:
                p = F.softmax(logits, dim=-1)
                k = p.shape[-1]
                hinge = []
                for row, i in enumerate(idx):
                    peers = [torch.tensor(np.pad(v, (0, k - len(v))), dtype=p.dtype, device=p.device)
                             for v in peer_probs[i].values()]
                    dists = [p[row]] + peers
                    js = [torch_js_normalized(dists[a], dists[b])
                          for a in range(len(dists)) for b in range(a + 1, len(dists))]
                    hinge.append(torch.relu(tau_js - torch.stack(js).mean()))
                div = lambda_div * torch.stack(hinge).mean()
            loss = task + div
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            logs["loss"].append(float(loss.item()))
            logs["task"].append(float(task.item()))
            logs["div"].append(float(div.item()))
        return {k: float(np.mean(v)) for k, v in logs.items()} | {"steps": steps, "n": len(questions)}

    def save(self, agent: str, path: Path) -> None:
        model = self.models[self.model_of[agent]]
        path = Path(path)
        model.save_pretrained(str(path), selected_adapters=[self._adapter_name(agent)])
        (path / "temperature.json").write_text(json.dumps({"temperature": self.temperature[agent]}))
