"""Table 1 random control on the test split, scored with the same composite D(q) and metrics as the real agents.

--mode letters    each agent picks a uniformly random letter with a
                  uniform(0,1) confidence; the option distribution is the max-entropy reconstruction of App. D.4
                  (confidence on the chosen letter, the rest spread evenly), as for letter-only agents.
--mode dirichlet  each agent emits a Dirichlet(1) option distribution.
--mode onehot     (default; matches the paper's mean D 0.732) each agent casts a uniformly random letter vote with confidence 1 (one-hot distribution).
One output dir per seed."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ddd")); sys.path.insert(0, str(ROOT / "repro"))
from coevolution import evaluate, score_questions  # noqa: E402
from config import load_config  # noqa: E402
from score_agents import load_split  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
p.add_argument("--out", required=True)
p.add_argument("--mode", choices=["letters", "dirichlet", "onehot"], default="onehot")
a = p.parse_args()
config = load_config(None, {})
qs = load_split(config, "test")
for seed in a.seeds:
    rng = np.random.default_rng(seed)
    def draw(k):
        if a.mode == "dirichlet":
            return rng.dirichlet(np.ones(k))
        c, j = (1.0 if a.mode == "onehot" else rng.uniform(0.0, 1.0)), rng.integers(k)
        v = np.full(k, (1.0 - c) / (k - 1)); v[j] = c
        return v
    probs = {ag: [draw(len(q["options"])) for q in qs] for ag in "RDC"}
    m = evaluate(score_questions(qs, probs, config["score"]["weights"]), "RDC",
                 config["eval"]["ece_bins"], config["score"]["hard_percentile"])
    d = Path(a.out) / a.mode / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"[random] seed {seed}: rho_D={m['rho_D']:.4f} mean_D={m['mean_D']:.4f} hard={m['hard_rate']:.3f}")
