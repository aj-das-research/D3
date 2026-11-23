"""Build repro/tracker.xlsx from the paper targets and the measured results.

Inputs (all in repro/):
  targets.csv     one row per numeric claim in the paper (id, paper_item, row, metric, paper_mean, paper_std, direction)
  results.csv     one row per measurement (id, value, std, n_seeds, run_tag, command, output_path, date, deviation_id, notes);
                  an id may appear several times, the LAST row is the current value, all rows go to the Runs sheet
  figures.csv     one row per paper figure with its reproduction status
  deviations.csv  one row per change to the paper's method or protocol (deviation_id, date, what, why, affects)

Verdict per row (one word):
  direction=higher : Better if value > paper + tol, Matched if |value - paper| <= tol, Below otherwise
  direction=lower  : mirrored
  direction=match  : Matched if |value - paper| <= tol, Mismatch otherwise
  no measurement   : Pending
  tol = paper_std when the paper reports one, else max(0.02, 5% of |paper|).

Usage:  python repro/build_tracker.py   (fails loudly on unknown ids or malformed rows)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
FILL = {
    "Better": PatternFill("solid", fgColor="A9D08E"),
    "Matched": PatternFill("solid", fgColor="C6EFCE"),
    "Below": PatternFill("solid", fgColor="FFC7CE"),
    "Mismatch": PatternFill("solid", fgColor="FFC7CE"),
    "Pending": PatternFill("solid", fgColor="EDEDED"),
    "Done": PatternFill("solid", fgColor="C6EFCE"),
    "Partial": PatternFill("solid", fgColor="FFEB9C"),
}
HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(color="FFFFFF", bold=True)


def tolerance(paper: float, std: float) -> float:
    if pd.notna(std) and std > 0:
        return float(std)
    return max(0.02, 0.05 * abs(paper))


def verdict(direction: str, paper: float, std: float, value: float) -> str:
    if pd.isna(value):
        return "Pending"
    tol = tolerance(paper, std)
    diff = value - paper
    if direction == "match":
        return "Matched" if abs(diff) <= tol else "Mismatch"
    if direction == "lower":
        diff = -diff
    elif direction != "higher":
        raise ValueError(f"unknown direction {direction!r}")
    if diff > tol:
        return "Better"
    if diff >= -tol:
        return "Matched"
    return "Below"


def read_csv(name: str, required: list[str]) -> pd.DataFrame:
    path = HERE / name
    df = pd.read_csv(path, dtype={"id": str})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[tracker] {path} is missing columns {missing}")
    print(f"[tracker] read {len(df)} rows from {path}")
    return df


def write_sheet(ws, df: pd.DataFrame, verdict_col: str | None = None, widths: dict[str, int] | None = None) -> None:
    ws.append(list(df.columns))
    for cell in ws[1]:
        cell.fill, cell.font = HDR_FILL, HDR_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in df.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
    if verdict_col is not None:
        col = list(df.columns).index(verdict_col) + 1
        for r in range(2, ws.max_row + 1):
            c = ws.cell(row=r, column=col)
            if c.value in FILL:
                c.fill = FILL[c.value]
                c.font = Font(bold=True)
    for i, name in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(name, max(10, min(40, len(str(name)) + 4)))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def main() -> None:
    targets = read_csv("targets.csv", ["id", "paper_item", "row", "metric", "paper_mean", "paper_std", "direction"])
    results = read_csv("results.csv", ["id", "value", "std", "n_seeds", "run_tag", "command", "output_path", "date",
                                       "deviation_id", "notes"])
    figures = read_csv("figures.csv", ["fig_id", "paper_file", "paper_ref", "description", "repro_command",
                                       "repro_output", "status", "notes"])
    deviations = read_csv("deviations.csv", ["deviation_id", "date", "what", "why", "affects"])

    if targets["id"].duplicated().any():
        raise SystemExit(f"[tracker] duplicate target ids: {targets.loc[targets['id'].duplicated(), 'id'].tolist()}")
    unknown = sorted(set(results["id"]) - set(targets["id"]))
    if unknown:
        raise SystemExit(f"[tracker] results.csv has ids not in targets.csv: {unknown}")
    known_dev = set(deviations["deviation_id"])
    used_dev = {p.strip() for d in results["deviation_id"].dropna() for p in str(d).split(";")}
    bad_dev = sorted(used_dev - known_dev)
    if bad_dev:
        raise SystemExit(f"[tracker] results.csv references unknown deviation ids: {bad_dev}")

    latest = results.drop_duplicates("id", keep="last").set_index("id")
    rows = []
    for t in targets.itertuples(index=False):
        r = latest.loc[t.id] if t.id in latest.index else None
        value = float(r["value"]) if r is not None else float("nan")
        std = float(r["std"]) if r is not None and pd.notna(r["std"]) else float("nan")
        rows.append({
            "id": t.id, "paper_item": t.paper_item, "row": t.row, "metric": t.metric,
            "paper": t.paper_mean, "paper_sd": t.paper_std, "direction": t.direction,
            "reproduced": round(value, 4) if pd.notna(value) else float("nan"),
            "repro_sd": round(std, 4) if pd.notna(std) else float("nan"),
            "n_seeds": r["n_seeds"] if r is not None else float("nan"),
            "delta": round(value - t.paper_mean, 4) if pd.notna(value) else float("nan"),
            "verdict": verdict(t.direction, t.paper_mean, t.paper_std, value),
            "deviation_id": r["deviation_id"] if r is not None else float("nan"),
            "run_tag": r["run_tag"] if r is not None else float("nan"),
            "output_path": r["output_path"] if r is not None else float("nan"),
            "notes": t.notes,
        })
    table = pd.DataFrame(rows)

    order = ["Better", "Matched", "Below", "Mismatch", "Pending"]
    summary = (table.groupby("paper_item", sort=False)["verdict"].value_counts().unstack(fill_value=0)
               .reindex(columns=order, fill_value=0).reset_index())
    summary["rows"] = summary[order].sum(axis=1)
    total = summary[order].sum().to_dict()
    summary.loc[len(summary)] = {"paper_item": "TOTAL", **total, "rows": sum(total.values())}

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    write_sheet(ws, summary, widths={"paper_item": 34})
    write_sheet(wb.create_sheet("Tables"), table, "verdict",
                {"id": 22, "paper_item": 28, "row": 36, "metric": 22, "output_path": 50, "notes": 30})
    write_sheet(wb.create_sheet("Figures"), figures, "status",
                {"paper_file": 34, "description": 50, "repro_command": 60, "repro_output": 50, "notes": 40})
    write_sheet(wb.create_sheet("Runs"), results, None,
                {"command": 80, "output_path": 50, "notes": 40, "run_tag": 26})
    write_sheet(wb.create_sheet("Deviations"), deviations, None, {"what": 70, "why": 70, "affects": 30})
    legend = pd.DataFrame([
        {"field": "verdict", "meaning": "Better / Matched / Below (direction higher|lower); Matched / Mismatch (direction match); Pending = not yet measured"},
        {"field": "tolerance", "meaning": "paper sd when reported, else max(0.02, 5% of |paper|)"},
        {"field": "source of truth", "meaning": "repro/targets.csv (paper), repro/results.csv (measurements), repro/figures.csv, repro/deviations.csv; rebuild with python repro/build_tracker.py"},
        {"field": "rule", "meaning": "every reproduced value is produced by the command and output_path in the Runs sheet; nothing is typed in by hand"},
    ])
    write_sheet(wb.create_sheet("Legend"), legend, None, {"field": 18, "meaning": 120})

    out = HERE / "tracker.xlsx"
    wb.save(out)
    print(f"[tracker] wrote {out}: {len(table)} rows, " + ", ".join(f"{k}={v}" for k, v in total.items()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
