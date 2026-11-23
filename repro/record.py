"""Append measured values to repro/results.csv, read programmatically from run outputs.

Every value is extracted from an output file; nothing is typed in by hand.

Sources (exactly one per call):
  --csv PATH --col COL [--where COL=VAL ...]       single cell from a CSV (the filters must select exactly one row)
  --json PATH --key a.b.c                          one value from a JSON file (dotted path)
  --jsonl PATH --key a.b --select last|max|...     one record from a JSONL log (default: last record)
  --glob PATTERN --key a.b                         JSON files matched by a glob (one per seed): mean and sample sd

Example:
  python repro/record.py --id T1.same50.rho --csv results/error_structure/rescored_summary.csv \
      --where file=heterogeneous_same_family_n50.jsonl --col spearman_rho_revised \
      --run_tag rescore_released --command "node analysis/error_structure.js"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COLUMNS = ["id", "value", "std", "n_seeds", "run_tag", "command", "output_path", "date", "deviation_id", "notes"]


def dig(obj, key: str):
    for part in key.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            if part not in obj:
                raise KeyError(f"key {part!r} (of {key!r}) not found; available: {list(obj)[:30]}")
            obj = obj[part]
    return float(obj)


def from_csv(path: str, col: str, where: list[str]) -> float:
    df = pd.read_csv(path)
    for w in where:
        k, v = w.split("=", 1)
        df = df[df[k].astype(str) == v]
    if len(df) != 1:
        raise SystemExit(f"[record] {path} with {where} selects {len(df)} rows, expected exactly 1")
    return float(df.iloc[0][col])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--csv")
    p.add_argument("--col")
    p.add_argument("--where", nargs="*", default=[])
    p.add_argument("--json")
    p.add_argument("--jsonl")
    p.add_argument("--select", default="last", help="jsonl record: last or iteration=N")
    p.add_argument("--glob")
    p.add_argument("--key")
    p.add_argument("--std_csv_col", help="CSV column holding the sd for the same row")
    p.add_argument("--run_tag", required=True)
    p.add_argument("--command", required=True, help="command that produced the output file")
    p.add_argument("--deviation_id", default="")
    p.add_argument("--notes", default="")
    a = p.parse_args()

    targets = pd.read_csv(HERE / "targets.csv", dtype={"id": str})
    if a.id not in set(targets["id"]):
        raise SystemExit(f"[record] unknown target id {a.id!r}")

    std, n_seeds = float("nan"), 1
    if a.csv:
        value = from_csv(a.csv, a.col, a.where)
        if a.std_csv_col:
            std = from_csv(a.csv, a.std_csv_col, a.where)
        out = a.csv
    elif a.json:
        value = dig(json.loads(Path(a.json).read_text()), a.key)
        out = a.json
    elif a.jsonl:
        recs = [json.loads(l) for l in Path(a.jsonl).read_text().splitlines() if l.strip()]
        if a.select == "last":
            rec = recs[-1]
        else:
            k, v = a.select.split("=", 1)
            hits = [r for r in recs if str(r.get(k)) == v]
            if len(hits) != 1:
                raise SystemExit(f"[record] {a.jsonl}: {a.select} matches {len(hits)} records")
            rec = hits[0]
        value = dig(rec, a.key)
        out = a.jsonl
    elif a.glob:
        files = sorted(glob.glob(a.glob))
        if not files:
            raise SystemExit(f"[record] glob {a.glob!r} matched nothing")
        vals = []
        for f in files:
            if f.endswith(".jsonl"):
                recs = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
                if a.select == "last":
                    rec = recs[-1]
                else:
                    k, v = a.select.split("=", 1)
                    hits = [r for r in recs if str(r.get(k)) == v]
                    if len(hits) != 1:
                        raise SystemExit(f"[record] {f}: {a.select} matches {len(hits)} records")
                    rec = hits[0]
                vals.append(dig(rec, a.key))
            else:
                vals.append(dig(json.loads(Path(f).read_text()), a.key))
        value = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        n_seeds = len(vals)
        out = a.glob
        print(f"[record] {a.id}: per-file values {dict(zip(files, np.round(vals, 4)))}")
    else:
        raise SystemExit("[record] give one of --csv, --json, --jsonl, --glob")

    if not np.isfinite(value):
        raise SystemExit(f"[record] {a.id}: extracted value is not finite ({value})")
    row = {"id": a.id, "value": value, "std": std, "n_seeds": n_seeds, "run_tag": a.run_tag, "command": a.command,
           "output_path": out, "date": dt.datetime.now().isoformat(timespec="seconds"),
           "deviation_id": a.deviation_id, "notes": a.notes}
    path = HERE / "results.csv"
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n").writerow(row)
    print(f"[record] {a.id} = {value:.4f}" + (f" +/- {std:.4f} (n={n_seeds})" if np.isfinite(std) else ""))


if __name__ == "__main__":
    main()
