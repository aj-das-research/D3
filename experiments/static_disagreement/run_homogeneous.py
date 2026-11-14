"""
Static disagreement: homogeneous role-prompted baseline
(paper Sections 5 and 6.1, Table 1).

Runs three copies of the SAME hosted model under different role prompts on
the MMLU-Pro hard subset, computes D(q) per question with the
DisagreementScorer, feeds hard cases into the HardCaseBuffer (dry run), and
measures the Spearman rank correlation between D(q) and the structural
complexity proxy stored in the `difficulty_score` field.

Default backend: NVIDIA NIM, meta/llama-3.3-70b-instruct.
Other OpenAI-compatible backends are selected with --backend.
API keys are read from the environment (see .env.example).
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
# sys.path : make the ddd/ modules importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ddd"))

from disagreement_scorer import DisagreementScorer, AgentOutput
from hard_case_buffer import HardCaseBuffer
from split_guard import select_records


# =====================================================================
# CONFIG - selected from CLI in __main__
# =====================================================================
BACKEND_CONFIGS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "model": "meta/llama-3.3-70b-instruct",
        "min_interval": 0.8,
        "json_mode": False,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "min_interval": 2.0,
        "json_mode": True,
    },
    "openai": {
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "min_interval": 0.0,
        "json_mode": True,
    },
}

BACKEND = "nvidia"
BASE_URL = None
API_KEY = None
MODEL = None
MIN_INTERVAL = 0.0
USE_JSON_MODE = False
client = None


def configure_backend(
    backend: str,
    model: Optional[str] = None,
    min_interval: Optional[float] = None,
) -> None:
    """Configure the OpenAI-compatible client from CLI args and env vars."""
    global BACKEND, BASE_URL, API_KEY, MODEL, MIN_INTERVAL, USE_JSON_MODE, client

    if backend not in BACKEND_CONFIGS:
        raise ValueError(f"Unknown backend: {backend}")

    cfg = BACKEND_CONFIGS[backend]
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"Missing {cfg['env_key']}. Set it first, for example:\n"
            f"  $env:{cfg['env_key']}='your_key_here'"
        )

    BACKEND = backend
    BASE_URL = cfg["base_url"]
    API_KEY = api_key
    MODEL = model or cfg["model"]
    MIN_INTERVAL = cfg["min_interval"] if min_interval is None else min_interval
    USE_JSON_MODE = cfg["json_mode"]
    client = (
        OpenAI(base_url=BASE_URL, api_key=API_KEY) if BASE_URL
        else OpenAI(api_key=API_KEY)
    )

DATA_PATH = PROJECT_ROOT / "data" / "mmlu_pro_hard_subset.csv"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "static_disagreement" / "outputs" / "homogeneous.jsonl"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
FAILURES_LOG = RESULTS_PATH.parent / "parse_failures.log"


def configure_output_paths(results_path: Optional[str] = None) -> None:
    global RESULTS_PATH, FAILURES_LOG
    if results_path:
        RESULTS_PATH = Path(results_path)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_LOG = RESULTS_PATH.parent / f"{RESULTS_PATH.stem}_parse_failures.log"

TEMPERATURE = 0.7    # some stochasticity so the three role prompts actually diverge
TOP_P       = 0.9
MAX_TOKENS  = 1024   # bumped from 512 -- MMLU-Pro hard questions are long,
                     #                   Llama's reasoning can eat tokens

HARD_THRESHOLD = 0.5    # matches the scorer default


# =====================================================================
# Role prompts + their mapping to agent-name letters.
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

BUNDLED_PROMPT_TEMPLATE = """Answer the following multiple-choice question three times, once for each role.

Question:
{question}

Options:
{options}

Roles:
- careful_reasoner: You are a careful reasoner. Work through the problem step by step, check your logic, and only commit to an answer once you are sure.
- domain_expert: You are a domain expert with deep specialised knowledge in this field. Apply your expertise directly and answer with authority.
- uncertain_analyst: You are a cautious analyst keenly aware of what you don't know. Acknowledge uncertainty where it exists; do not overstate confidence.

Respond ONLY with a JSON object in this exact format -- no prose, no code fences:
{{
  "careful_reasoner": {{"answer": "<letter>", "confidence": <float in [0,1]>}},
  "domain_expert": {{"answer": "<letter>", "confidence": <float in [0,1]>}},
  "uncertain_analyst": {{"answer": "<letter>", "confidence": <float in [0,1]>}}
}}"""


# =====================================================================
# Naive rate limiter
# =====================================================================
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


def parse_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:idx + 1])
                    except json.JSONDecodeError:
                        return None
        return None


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

BUNDLED_SYSTEM_MSG = (
    "You are a JSON-only response system. "
    "Output exactly one JSON object and nothing else. "
    "No prose before it, no prose after it, no code fences, no markdown. "
    "The JSON must have exactly three top-level keys: careful_reasoner, "
    "domain_expert, and uncertain_analyst. Each value must contain answer "
    "and confidence."
)

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
    if client is None:
        raise RuntimeError("API client is not configured. Call configure_backend first.")

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


def call_all_roles_bundled(
    question: str,
    options_str: str,
    qid: str = "?",
    retries: int = 3,
) -> dict:
    """One API call returns all three homogeneous role outputs."""
    if client is None:
        raise RuntimeError("API client is not configured. Call configure_backend first.")

    user_prompt = BUNDLED_PROMPT_TEMPLATE.format(
        question=question,
        options=options_str,
    )
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": BUNDLED_SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=512,
    )
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    last_raw = ""
    for attempt in range(retries):
        try:
            _rate_limit()
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            last_raw = text
            obj = parse_json_object(text)
            if isinstance(obj, dict):
                parsed = {}
                ok = True
                for role_name, _, _ in ROLES:
                    role_obj = obj.get(role_name, {})
                    ans = str(role_obj.get("answer", "")).strip().upper()[:1] or None
                    try:
                        conf = float(role_obj.get("confidence"))
                        conf = max(0.0, min(1.0, conf))
                    except (TypeError, ValueError):
                        conf = None
                    parsed[role_name] = {
                        "answer": ans,
                        "confidence": conf,
                        "raw": json.dumps(role_obj),
                    }
                    ok = ok and ans is not None and conf is not None
                if ok:
                    return parsed
            _log_failure("bundled_roles", qid, attempt, text)
        except Exception as e:
            wait = 2 ** attempt
            print(f"\n[retry {attempt+1}/{retries}] {type(e).__name__}: {e} "
                  f"-- sleeping {wait}s", flush=True)
            time.sleep(wait)

    return {
        role_name: {"answer": None, "confidence": None, "raw": last_raw}
        for role_name, _, _ in ROLES
    }


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
def main(
    limit: Optional[int] = None,
    bundle_roles: bool = False,
    split_manifest: Optional[str] = None,
    split_manifest_sha256: Optional[str] = None,
    eval_split: str = "validation",
):
    # ---- load MMLU-Pro hard subset (prepared CSV) ----------------------
    # We load from CSV not JSON because the JSON serializer wrote the
    # CAPPED `final_difficulty` values into a field named `difficulty_score`,
    # causing a ceiling at 1.0 that kills Spearman variance. The CSV keeps
    # the raw uncapped `difficulty_score` column intact.
    import ast
    df_in = pd.read_csv(DATA_PATH)
    # `options` column is a stringified Python list -- parse it back
    df_in["options"] = df_in["options"].apply(ast.literal_eval)
    questions = df_in.to_dict(orient="records")
    questions, split_report = select_records(
        questions,
        split_manifest
        or (
            PROJECT_ROOT
            / "data"
            / "splits"
            / "mmlu_pro_hard_split_manifest.csv"
        ),
        {eval_split},
        expected_sha256=split_manifest_sha256,
    )
    if limit:
        questions = questions[:limit]
    print(
        f"Loaded {len(questions)} {eval_split} questions from {DATA_PATH.name} "
        f"(manifest {split_report['manifest_sha256'][:12]})"
    )
    print(f"  difficulty_score range: "
          f"{df_in['difficulty_score'].min():.3f} -- "
          f"{df_in['difficulty_score'].max():.3f}  "
          f"(unique: {df_in['difficulty_score'].nunique()})")

    done = load_already_done()
    if done:
        print(f"Resuming: {len(done)} already complete, skipping those")

    # ---- instantiate the scorer and the buffer -----------------------
    scorer = DisagreementScorer(threshold=HARD_THRESHOLD)
    buffer = HardCaseBuffer(max_size=500, sample_size=200)

    # ---- main loop ---------------------------------------------------
    with RESULTS_PATH.open("a", encoding="utf-8") as fout:
        for q in tqdm(questions, desc="Homogeneous baseline"):
            qid = str(q["question_id"])
            if qid in done:
                continue

            opts_str = format_options(q["options"])

            # Call the three role-prompted agents. Bundled mode is much faster:
            # one API call per question instead of three.
            per_role = {}
            agent_outputs = []
            bundled = (
                call_all_roles_bundled(q["question"], opts_str, qid=qid)
                if bundle_roles
                else None
            )
            for role_name, agent_letter, role_prompt in ROLES:
                if bundled is None:
                    out = call_agent(
                        role_prompt, q["question"], opts_str,
                        role_name=role_name, qid=qid,
                    )
                else:
                    out = bundled[role_name]
                per_role[role_name] = {
                    "agent_letter": agent_letter,
                    "answer":       out["answer"],
                    "confidence":   out["confidence"],
                    "raw":          out["raw"],
                }
                # Feed the scorer. If parsing failed we fall back
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

            # ---- disagreement scoring ------------------------------------
            result = scorer.score(qid, q["question"], agent_outputs)

            # ---- hard case buffer (dry-run) ------------------------------
            # iteration=0 because the static run is not the co-evolution loop
            buffer.add(result, iteration=0, c_wrong_conf=None)

            # ---- write row -----------------------------------------------
            rec = {
                "question_id":      qid,
                "protocol_split":   q["protocol_split"],
                "category":         q.get("category"),
                "difficulty_score": q.get("difficulty_score"),
                "correct_answer":   q.get("answer"),      # letter
                "per_role":         per_role,
                "D_answer":         result.d_answer,
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
    print("Homogeneous role-prompted baseline (Table 1)")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_CONFIGS),
        default="nvidia",
        help="OpenAI-compatible provider to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider default model.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=None,
        help="Minimum seconds between API calls. Lower is faster but may rate-limit.",
    )
    parser.add_argument(
        "--bundle-roles",
        action="store_true",
        help="Use one API call per question for all three role prompts.",
    )
    parser.add_argument(
        "--results-path",
        default=None,
        help="Write/read resume JSONL at this path.",
    )
    parser.add_argument(
        "--split-manifest",
        default=str(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "mmlu_pro_hard_split_manifest.csv"
        ),
        help="Immutable question_id/split CSV. Evaluation is fail-closed.",
    )
    parser.add_argument(
        "--split-manifest-sha256",
        default=os.environ.get("SPLIT_MANIFEST_SHA256"),
    )
    parser.add_argument(
        "--eval-split",
        choices=["validation", "test"],
        default="validation",
        help="Use validation while developing; use test once for final reporting.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run on first N questions only (use --limit 10 for smoke test)",
    )
    args = parser.parse_args()
    default_results_path = (
        PROJECT_ROOT
        / "experiments"
        / "static_disagreement"
        / "outputs"
        / f"homogeneous_{args.eval_split}.jsonl"
    )
    configure_output_paths(args.results_path or str(default_results_path))
    configure_backend(
        backend=args.backend,
        model=args.model,
        min_interval=args.min_interval,
    )
    main(
        limit=args.limit,
        bundle_roles=args.bundle_roles,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        eval_split=args.eval_split,
    )
