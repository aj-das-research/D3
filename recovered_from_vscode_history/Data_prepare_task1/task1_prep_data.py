"""
TASK 1 — Download & Prep MMLU-Pro Hard Subset
Run in: VS Code / local
Install first: pip install datasets pandas numpy scipy
"""

import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import entropy

# ── 1. Load MMLU-Pro ──────────────────────────────────────────────────────────
print("Loading MMLU-Pro dataset...")
dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
df = pd.DataFrame(dataset)

print(f"Total questions: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Categories: {df['category'].unique()}")


# ── 2. Compute Difficulty Score Per Question ──────────────────────────────────
# MMLU-Pro has 10 answer choices (vs MMLU's 4).
# We use a proxy for difficulty: questions where the answer distribution
# (if available) is most uniform = hardest for annotators to agree on.
# Since MMLU-Pro doesn't ship annotator confidence directly, we use two proxies:
#   (a) Number of answer options (always 10 in MMLU-Pro — so same for all)
#   (b) Category-level hardness (known from literature)
#   (c) Question length + option similarity as a surface proxy

def compute_surface_difficulty(row):
    """
    Proxy difficulty score based on:
    - Question length (longer = harder, generally)
    - Average answer option length (more content = harder to distinguish)
    - Number of numeric/formula tokens (math-heavy = harder)
    """
    question = str(row["question"])
    options = row["options"]  # list of 10 strings

    # Feature 1: question length (normalized)
    q_len = len(question.split())

    # Feature 2: mean option length
    opt_len = np.mean([len(str(o).split()) for o in options])

    # Feature 3: math token count (digits, symbols)
    math_tokens = sum(1 for c in question if c in "0123456789=+−×÷^√∫∑")

    # Feature 4: option text similarity (higher similarity = harder to distinguish)
    # Simple proxy: std of option lengths (low std = similar-length = harder)
    opt_len_std = np.std([len(str(o).split()) for o in options])
    similarity_score = 1 / (opt_len_std + 1)  # higher = more similar options

    # Combine into a single difficulty score (higher = harder)
    difficulty = (
        0.3 * min(q_len / 100, 1.0) +        # question length, capped
        0.2 * min(opt_len / 30, 1.0) +        # option length
        0.3 * min(math_tokens / 20, 1.0) +    # math density
        0.2 * similarity_score                  # option similarity
    )
    return round(difficulty, 4)


print("\nComputing difficulty scores...")
df["difficulty_score"] = df.apply(compute_surface_difficulty, axis=1)


# ── 3. Category-Level Hardness Boost ─────────────────────────────────────────
# From literature, these categories are consistently hardest on MMLU-Pro
HARD_CATEGORIES = {
    "math":        0.20,
    "physics":     0.15,
    "chemistry":   0.12,
    "engineering": 0.10,
    "law":         0.08,
    "biology":     0.05,
}

def category_boost(category):
    cat = str(category).lower()
    for key, boost in HARD_CATEGORIES.items():
        if key in cat:
            return boost
    return 0.0

df["category_boost"] = df["category"].apply(category_boost)
df["final_difficulty"] = (df["difficulty_score"] + df["category_boost"]).clip(0, 1)
df["final_difficulty"] = df["final_difficulty"].round(4)


# ── 4. Extract Hard Subset ────────────────────────────────────────────────────
# Hard subset = top 25% by final_difficulty score
threshold = df["final_difficulty"].quantile(0.75)
hard_df = df[df["final_difficulty"] >= threshold].copy()
hard_df = hard_df.sort_values("final_difficulty", ascending=False).reset_index(drop=True)

print(f"\nDifficulty threshold (75th percentile): {threshold:.4f}")
print(f"Hard subset size: {len(hard_df)} questions")
print(f"Category distribution in hard subset:")
print(hard_df["category"].value_counts().to_string())


# ── 5. Save Outputs ───────────────────────────────────────────────────────────

# Full dataset with difficulty scores
df.to_csv("mmlu_pro_full_with_difficulty.csv", index=False)
print("\nSaved: mmlu_pro_full_with_difficulty.csv")

# Hard subset as CSV
hard_df.to_csv("mmlu_pro_hard_subset.csv", index=False)
print("Saved: mmlu_pro_hard_subset.csv")

# Hard subset as JSON (easier to load in later tasks)
hard_records = []
for _, row in hard_df.iterrows():
    hard_records.append({
        "question_id":      str(row.get("question_id", row.name)),
        "question":         row["question"],
        "options":          list(row["options"]),
        "answer":           row["answer"],           # correct answer letter
        "answer_index":     int(row["answer_index"]) if "answer_index" in row else -1,
        "category":         row["category"],
        "difficulty_score": row["final_difficulty"],
    })

with open("mmlu_pro_hard_subset.json", "w") as f:
    json.dump(hard_records, f, indent=2)
print("Saved: mmlu_pro_hard_subset.json")

# Also save a small sample (50 questions) for quick testing in later tasks
sample = hard_records[:50]
with open("mmlu_pro_hard_sample_50.json", "w") as f:
    json.dump(sample, f, indent=2)
print("Saved: mmlu_pro_hard_sample_50.json  (50-question sample for quick testing)")


# ── 6. Sanity Check ───────────────────────────────────────────────────────────
print("\n── SANITY CHECK ──────────────────────────────────────────────────────")
print(f"Hard subset shape:        {hard_df.shape}")
print(f"Difficulty range:         {hard_df['final_difficulty'].min():.4f} → {hard_df['final_difficulty'].max():.4f}")
print(f"Mean difficulty:          {hard_df['final_difficulty'].mean():.4f}")
print(f"\nSample hard question:")
sample_q = hard_records[0]
print(f"  Category:    {sample_q['category']}")
print(f"  Difficulty:  {sample_q['difficulty_score']}")
print(f"  Question:    {sample_q['question'][:120]}...")
print(f"  # Options:   {len(sample_q['options'])}")
print("\n✅ Task 1 complete. Files ready for Task 2.")