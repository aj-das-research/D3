"""
Draw the paper's result figures from run outputs.

Figure 2  Co-evolution dynamics: per-agent accuracy, rho(D, s), Agent-R ECE,
          mean pairwise JS with and without the regulariser (mean +/- 1 sd
          over seeds). Reads every `iterations.jsonl` under --coevolution_dir.
Figure 3  Per-agent accuracy, ECE and unique-correct counts, from
          results/error_structure/orthogonality_summary.json.
Figure 5  Spearman rho and mean disagreement per static setting, from
          results/error_structure/rescored_summary.json.

Usage:
    python analysis/plot_figures.py --coevolution_dir outputs/coevolution --out results/figures
Figures whose inputs are missing are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_COLORS = {"R": "#2a6fdb", "D": "#e08a1e", "C": "#2e9d5b"}


def load_iterations(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(root).rglob("iterations.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            df = pd.DataFrame(rows)
            df["run"] = str(path.parent)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _band(ax, df, col, label, color=None):
    g = df.groupby("iteration")[col]
    m, s = g.mean(), g.std().fillna(0)
    ax.plot(m.index, m.values, label=label, color=color)
    ax.fill_between(m.index, m - s, m + s, alpha=0.2, color=color)


def figure2(df: pd.DataFrame, out: Path) -> None:
    reg = df[df["regularized"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for a, c in AGENT_COLORS.items():
        if f"acc_{a}" in reg:
            _band(axes[0, 0], reg, f"acc_{a}", f"Agent-{a}", c)
            _band(axes[1, 0], reg, f"ece_{a}", f"Agent-{a}", c)
    _band(axes[0, 1], reg, "rho_D", "D", "#444")
    _band(axes[0, 1], reg, "rho_D_LF", "D_LF", "#999")
    _band(axes[1, 1], reg, "mean_js", "With regulariser", "#2a6fdb")
    noreg = df[~df["regularized"]]
    if len(noreg):
        _band(axes[1, 1], noreg, "mean_js", "Without regulariser", "#d1495b")
    titles = ["(a) Per-agent accuracy", "(b) Spearman rho(D, s)", "(c) ECE", "(d) Mean pairwise JS"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t)
        ax.set_xlabel("Co-evolution iteration")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure2_coevolution.pdf")
    fig.savefig(out / "figure2_coevolution.png", dpi=200)
    plt.close(fig)


def figure3(summary: dict, out: Path) -> None:
    agents = ["R", "D", "C"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    colors = [AGENT_COLORS[a] for a in agents]
    for ax, key, title in ((axes[0], "accuracy", "(a) Accuracy"), (axes[1], "ECE", "(b) ECE (lower is better)")):
        vals = [summary[f"{a}_{key}"] for a in agents]
        ax.bar([f"Agent-{a}" for a in agents], vals, color=colors)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    uniq = [summary[f"{a}_only_correct"] for a in agents]
    axes[2].bar([f"{a} only" for a in agents], uniq, color=colors)
    axes[2].set_title("(c) Unique-correct cases")
    fig.tight_layout()
    fig.savefig(out / "figure3_per_agent.pdf")
    fig.savefig(out / "figure3_per_agent.png", dpi=200)
    plt.close(fig)


def figure5(rows: list, out: Path) -> None:
    rows = [r for r in rows if r.get("n_scored")]
    names = [r["file"].replace(".jsonl", "") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(names, [r["spearman_rho_revised"] for r in rows], color="#2a6fdb")
    axes[0].axhline(0, color="#444", lw=0.8)
    axes[0].set_title("(a) Spearman rho(D, s)")
    axes[1].bar(names, [r["mean_D_revised"] for r in rows], color="#e08a1e")
    axes[1].set_title("(b) Mean disagreement")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure5_static_settings.pdf")
    fig.savefig(out / "figure5_static_settings.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coevolution_dir", default=str(REPO_ROOT / "outputs" / "coevolution"))
    p.add_argument("--error_structure_dir", default=str(REPO_ROOT / "results" / "error_structure"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "figures"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_iterations(Path(args.coevolution_dir))
    if len(df):
        figure2(df, out)
        print("wrote figure 2")
    else:
        print(f"skip figure 2: no iterations.jsonl under {args.coevolution_dir}")

    es = Path(args.error_structure_dir)
    if (es / "orthogonality_summary.json").exists():
        s = json.loads((es / "orthogonality_summary.json").read_text())
        s = {**s, **{f"{a}_accuracy": s[f"{a}_accuracy"] for a in "RDC"}}
        figure3(s, out)
        print("wrote figure 3")
    if (es / "rescored_summary.json").exists():
        figure5(json.loads((es / "rescored_summary.json").read_text()), out)
        print("wrote figure 5")


if __name__ == "__main__":
    main()
