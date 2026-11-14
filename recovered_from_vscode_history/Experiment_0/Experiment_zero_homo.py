"""
Experiment 0 - Homogeneous Baseline  (Task 4 of coding checklist)
==================================================================

Runs 3 copies of the SAME model with different role prompts on the
MMLU-Pro hard subset, computes D(q) per question via your Task 2
DisagreementScorer, feeds hard cases into your Task 3 HardCaseBuffer
(dry-run), and measures Spearman rank correlation between D(q) and
the `difficulty_score` field produced in Task 1.

This is the HOMOGENEOUS row of Table 1 in your paper. The heterogeneous
row (Agent-R/D/C) is filled in later once Weeks 1-2 training is done.

TEST MODE (default): NVIDIA NIM, meta/llama-3.3-70b-instruct  -- free
REAL RUN:            flip BACKEND = "openai"  (one line change)

Folder layout assumed:
    D:\\Multi_Agent_Code\\
    |-- Data_prepare_task1\\mmlu_pro_hard_subset.json
    |-- Disagreement_scorer\\task2_disagreement_scorer.py
    |-- HARD_CASE_BUFFER_TASK3\\Task3_hard_case_buffer.py
    |-- Experiment_0\\Experiment_zero_homo.py    (this file)
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Tuple

from openai import OpenAI
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd


# ---------------------------------------------------------------------
# sys.path : make your Task 2 and Task 3 folders importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # D:\Multi_Agent_Code
sys.path.insert(0, str(PROJECT_ROOT / "Disagreement_scorer"))
sys.path.insert(0, str(PROJECT_ROOT / "HARD_CASE_BUFFER_TASK3"))

# Importing Task 2 triggers a ~80 MB sentence-transformers model download
# the first time, then ~20s to load. Be patient on the first run.
from task2_disagreement_scorer import DisagreementScorer, AgentOutput
from Task3_hard_case_buffer import HardCaseBuffer


# =====================================================================
# CONFIG  -- swap NVIDIA <-> GROQ <-> OpenAI here
# =====================================================================
BACKEND = "groq"          # "nvidia" | "groq" | "openai"

if BACKEND == "nvidia":
    BASE_URL      = "https://integrate.api.nvidia.com/v1"
    API_KEY       = os.environ["NVIDIA_API_KEY"]             # nvapi-...
    MODEL         = "meta/llama-3.3-70b-instruct"
    MIN_INTERVAL  = 1.6        # 40 req/min free-tier cap
    USE_JSON_MODE = False      # NIM support is model-dependent; skip
elif BACKEND == "groq":
    BASE_URL      = "https://api.groq.com/openai/v1"
    API_KEY       = "REDACTED_GROQ_API_KEY"               # gsk_...
    MODEL         = "llama-3.3-70b-versatile"
    MIN_INTERVAL  = 22.0       # TPM=6000 is the real cap; ~2-3 req/min
    USE_JSON_MODE = True       # Groq honours response_format properly
elif BACKEND == "openai":
    BASE_URL      = None       # default OpenAI endpoint
    API_KEY       = os.environ["OPENAI_API_KEY"]
    MODEL         = "gpt-4o-mini"
    MIN_INTERVAL  = 0.0        # OpenAI handles rate-limiting server-side
    USE_JSON_MODE = True
else:
    raise ValueError(f"Unknown BACKEND: {BACKEND}")

DATA_PATH    = PROJECT_ROOT / "Data_prepare_task1" / "mmlu_pro_hard_subset.csv"
RESULTS_PATH = PROJECT_ROOT / "Experiment_0" / "results" / "exp0_homogeneous.jsonl"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

TEMPERATURE = 0.7    # some stochasticity so the three role prompts actually diverge
TOP_P       = 0.9
MAX_TOKENS  = 1024   # bumped from 512 -- MMLU-Pro hard questions are long,
                     #                   Llama's reasoning can eat tokens

HARD_THRESHOLD = 0.5    # matches your Task 2 default


# =====================================================================
# Role prompts + their mapping to Task 2 agent-name letters.
# We use "R"/"D"/"C" internally so HardCaseBuffer._compute_rd_gap works
# unchanged during this homogeneous dry-run.
# =====================================================================
ROLES: List[Tuple[str, str, str]] = [
    ("careful_reasoner", "R",
     "You are a careful reasoner. Work through the problem step by step, "
     "check your logic, and only commit to an answer once you are sure."),
    ("domain_expert", "D",
     "You are a domain expert with deep specialised knowledge in this field. "
     "Apply your expertise directly and answer with authority."),
    ("uncertain_analyst", "C",
     "You are a cautious analyst keenly aware of what you don't know. "
     "Acknowledge uncertainty where it exists; do not overstate confidence."),
]

PROMPT_TEMPLATE = """{role_prompt}

Answer the following multiple-choice question.

Question:
{question}

Options:
{options}

Respond ONLY with a JSON object in this exact format -- no prose, no code fences:
{{"answer": "<letter>", "confidence": <float in [0,1]>}}

The "confidence" field is your honest self-rated probability that your answer is correct."""


# =====================================================================
# Client + naive rate limiter
# =====================================================================
client = (
    OpenAI(base_url=BASE_URL, api_key=API_KEY) if BASE_URL
    else OpenAI(api_key=API_KEY)
)

_last_call_t = [0.0]
def _rate_limit():
    delta = time.time() - _last_call_t[0]
    if delta < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - delta)
    _last_call_t[0] = time.time()


# =====================================================================
# Response parsing (models sometimes sneak in prose; be defensive)
# =====================================================================
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)

def parse_response(text: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (answer_letter, confidence) or (None, None) on failure."""
    if not text:
        return None, None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None, None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, None
    ans = str(obj.get("answer", "")).strip().upper()[:1] or None
    try:
        conf = float(obj.get("confidence"))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = None
    return ans, conf


# =====================================================================
# Single agent call (with retry + exponential backoff)
# =====================================================================
SYSTEM_MSG = (
    "You are a JSON-only response system. "
    "Output exactly one JSON object and nothing else. "
    "No prose before it, no prose after it, no code fences, no markdown. "
    "The JSON must have exactly two keys: \"answer\" (a single letter A-J) "
    "and \"confidence\" (a float between 0 and 1)."
)

FAILURES_LOG = RESULTS_PATH.parent / "parse_failures.log"


def _log_failure(role_name: str, qid: str, attempt: int, text: str):
    """Append a parse failure to the debug log so we can see what went wrong."""
    with FAILURES_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"qid={qid} role={role_name} attempt={attempt}\n")
        f.write(f"{'=' * 60}\n")
        f.write(text if text else "<EMPTY RESPONSE>")
        f.write("\n")


def call_agent(role_prompt: str, question: str, options_str: str,
               role_name: str = "?", qid: str = "?",
               retries: int = 3) -> dict:
    user_prompt = PROMPT_TEMPLATE.format(
        role_prompt=role_prompt, question=question, options=options_str,
    )
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    if USE_JSON_MODE:
        # Forces the server to return valid JSON. Kills the parse-drop issue.
        kwargs["response_format"] = {"type": "json_object"}

    last_raw = ""
    for attempt in range(retries):
        try:
            _rate_limit()
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            last_raw = text
            ans, conf = parse_response(text)
            if ans is not None and conf is not None:
                return {"answer": ans, "confidence": conf, "raw": text}
            # parse fail -- log and try again
            _log_failure(role_name, qid, attempt, text)
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n[retry {attempt+1}/{retries}] {type(e).__name__}: {e} "
                  f"-- sleeping {wait}s", flush=True)
            time.sleep(wait)
    return {"answer": None, "confidence": None, "raw": last_raw}


# =====================================================================
# Helpers
# =====================================================================
def format_options(options: List[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))


def load_already_done() -> set:
    """Resume support: read existing results and skip completed qids."""
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    with RESULTS_PATH.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["question_id"])
            except Exception:
                pass
    return done


# =====================================================================
# Main
# =====================================================================
def main(limit: Optional[int] = None):
    # ---- load MMLU-Pro hard subset (Task 1 CSV output) ---------------
    # We load from CSV not JSON because the JSON serializer wrote the
    # CAPPED `final_difficulty` values into a field named `difficulty_score`,
    # causing a ceiling at 1.0 that kills Spearman variance. The CSV keeps
    # the raw uncapped `difficulty_score` column intact.
    import ast
    df_in = pd.read_csv(DATA_PATH)
    # `options` column is a stringified Python list -- parse it back
    df_in["options"] = df_in["options"].apply(ast.literal_eval)
    questions = df_in.to_dict(orient="records")
    if limit:
        questions = questions[:limit]
    print(f"Loaded {len(questions)} questions from {DATA_PATH.name}")
    print(f"  difficulty_score range: "
          f"{df_in['difficulty_score'].min():.3f} -- "
          f"{df_in['difficulty_score'].max():.3f}  "
          f"(unique: {df_in['difficulty_score'].nunique()})")

    done = load_already_done()
    if done:
        print(f"Resuming: {len(done)} already complete, skipping those")

    # ---- instantiate your Task 2 scorer + Task 3 buffer --------------
    scorer = DisagreementScorer(threshold=HARD_THRESHOLD)
    buffer = HardCaseBuffer(max_size=500, sample_size=200)

    # ---- main loop ---------------------------------------------------
    with RESULTS_PATH.open("a", encoding="utf-8") as fout:
        for q in tqdm(questions, desc="Experiment 0"):
            qid = str(q["question_id"])
            if qid in done:
                continue

            opts_str = format_options(q["options"])

            # Call the three role-prompted agents
            per_role = {}
            agent_outputs = []
            for role_name, agent_letter, role_prompt in ROLES:
                out = call_agent(
                    role_prompt, q["question"], opts_str,
                    role_name=role_name, qid=qid,
                )
                per_role[role_name] = {
                    "agent_letter": agent_letter,
                    "answer":       out["answer"],
                    "confidence":   out["confidence"],
                    "raw":          out["raw"],
                }
                # Feed your Task 2 scorer. If parsing failed we fall back
                # to a placeholder so the call still succeeds; the record
                # is filtered out of Spearman below.
                agent_outputs.append(AgentOutput(
                    agent_name = agent_letter,
                    answer     = out["answer"] if out["answer"] is not None else "UNKNOWN",
                    confidence = out["confidence"] if out["confidence"] is not None else 0.0,
                ))

            any_parse_fail = any(
                per_role[r]["answer"] is None or per_role[r]["confidence"] is None
                for r, _, _ in ROLES
            )

            # ---- Task 2: disagreement scoring ----------------------------
            result = scorer.score(qid, q["question"], agent_outputs)

            # ---- Task 3: hard case buffer (dry-run) ----------------------
            # iteration=0 because Experiment 0 is not the co-evolution loop
            buffer.add(result, iteration=0, c_wrong_conf=None)

            # ---- write row -----------------------------------------------
            rec = {
                "question_id":      qid,
                "category":         q.get("category"),
                "difficulty_score": q.get("difficulty_score"),
                "correct_answer":   q.get("answer"),      # letter
                "per_role":         per_role,
                "D_semantic":       result.d_semantic,
                "D_confidence":     result.d_confidence,
                "D_q":              result.d_score,
                "is_hard_case":     result.is_hard_case,
                "parse_failed":     any_parse_fail,
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

    # =================================================================
    # Analysis -- this is what goes in Table 1 (homogeneous row)
    # =================================================================
    df = pd.read_json(RESULTS_PATH, lines=True)

    # Drop rows where any agent failed to parse a valid answer/confidence
    clean = df[~df["parse_failed"]].copy()
    clean = clean.dropna(subset=["D_q", "difficulty_score"])

    rho, p = spearmanr(clean["D_q"], clean["difficulty_score"])

    print("\n" + "=" * 64)
    print("EXPERIMENT 0 -- Homogeneous Baseline (Table 1 homogeneous row)")
    print("=" * 64)
    print(f"Backend:            {BACKEND}")
    print(f"Model:              {MODEL}")
    print(f"N questions total:  {len(df)}")
    print(f"N parseable:        {len(clean)}  "
          f"({len(df) - len(clean)} dropped)")
    if len(df) > 0:
        drop_rate = (len(df) - len(clean)) / len(df)
        print(f"Parse-drop rate:    {drop_rate:.1%}  "
              f"{'<-- investigate parse_failures.log' if drop_rate > 0.20 else ''}")
    print(f"Spearman rho:       {rho:+.4f}")
    print(f"p-value:            {p:.4e}")
    print(f"Significant?        {'yes' if p < 0.05 else 'no'}  (alpha=0.05)")
    print("-" * 64)

    # Buffer dry-run diagnostics
    print(f"Buffer size:        {len(buffer)} / 500")
    print(f"Buffer stats:       {buffer.stats()}")
    print("-" * 64)

    # Per-role accuracy (sanity check -- unrelated to the thesis)
    for role_name, letter, _ in ROLES:
        def is_correct(row):
            return (row["per_role"][role_name]["answer"] == row["correct_answer"])
        correct = clean.apply(is_correct, axis=1).sum()
        print(f"  {role_name:20s} ({letter}) acc: "
              f"{correct / max(len(clean), 1):.3f}")
    print("=" * 64)
    print()
    print("Interpretation (per plan Section 10):")
    print("  |rho| < 0.2 or p > 0.05 : homogeneous disagreement does NOT")
    print("                            track difficulty.  Thesis survives,")
    print("                            proceed to train Agent-R next.")
    print("  |rho| > 0.3, p < 0.05   : homogeneous already correlates well.")
    print("                            Pause and revisit the thesis before")
    print("                            burning GPU hours on training.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run on first N questions only (use --limit 10 for smoke test)",
    )
    args = parser.parse_args()
    main(limit=args.limit)