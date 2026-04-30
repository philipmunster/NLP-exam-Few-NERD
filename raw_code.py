  """
instructllama_test.py
─────────────────────────────────────────────────────────────────────────────
Llama 3.1 8B Instruct — ICL evaluation on Few-NERD (INTRA split)
─────────────────────────────────────────────────────────────────────────────

What this script does:
  1. Downloads Few-NERD (INTRA) from HuggingFace automatically
  2. Runs a mini prompt-comparison experiment (P1 vs P2 vs P3) on one config
  3. Picks the best prompt by Span-F1
  4. Runs all 8 INTRA configurations using the winning prompt
  5. Saves results to results/

Experiment grid (INTRA only = 8 configs + 3-prompt mini experiment):
  N-way:   5, 10
  K-shot:  low (1~2), high (5~10)
  Prompts: P1, P2, P3  (mini only) → best prompt used for full eval

Output files:
  results/mini_experiment.json          <- prompt comparison
  results/instructllama_results.json    <- all configs, metrics, predictions

─────────────────────────────────────────────────────────────────────────────
USAGE
  pip install requests datasets
  python instructllama_test.py

Make sure LM Studio is running with Llama 3.1 8B Instruct loaded.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import re
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  <- the only section you need to edit
# ─────────────────────────────────────────────────────────────────────────────

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
# ^ Change port if LM Studio is running elsewhere, e.g. http://localhost:8080/...

LM_STUDIO_MODEL = "meta-llama-3.1-8b-instruct"
# ^ Must match exactly the model name shown in LM Studio's model picker

RESULTS_DIR = Path("results")

# Episode counts
MINI_EPISODES = 5    # episodes per prompt in the mini experiment (fast)
FULL_EPISODES = 50   # episodes per config in the full evaluation
                     # Few-NERD paper uses 100 -- lower this if time is limited

# K-shot ranges (matching Few-NERD paper notation)
K_RANGES = {
    "low":  (1, 2),    # 1~2 shot
    "high": (5, 10),   # 5~10 shot
}

N_WAYS = [5, 10]

# Mini experiment runs on this one config before choosing the best prompt
MINI_CONFIG = {"n_way": 5, "k_range": "low"}

SEED = 42
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
#
# Three strategies covering the main design axes for ICL NER:
#
#   P1 - Minimal:
#        Bare-bones instruction. Tests whether the model can perform NER
#        from pure instruction-following with no explicit format scaffolding.
#        Hypothesis: likely to produce inconsistent output formats.
#
#   P2 - Structured ICL:
#        Explicit support-set demonstration with strict format rules.
#        Mirrors how prototype networks use support examples -- the model
#        sees labeled examples and must generalise to the query.
#        Hypothesis: best balance of accuracy and parse reliability.
#
#   P3 - Chain-of-thought + structured output:
#        Asks the model to reason before labeling. Tests whether explicit
#        reasoning improves span boundary detection, at the cost of latency
#        and more complex parsing.
#        Hypothesis: may help on ambiguous spans but slow and noisy to parse.

PROMPTS = {

    "P1": (
        "You are a named entity recognition system.\n"
        "Label each token in the query sentence using only the entity types provided.\n"
        "Output format: one token:::LABEL per line.\n"
        "Use O for non-entity tokens.\n"
        "Output nothing else -- no explanation, no extra text."
    ),

    "P2": (
        "You are an expert named entity recognition (NER) system.\n"
        "\n"
        "You will receive:\n"
        "  - SUPPORT EXAMPLES: sentences with every token already labeled.\n"
        "  - A QUERY sentence: unlabeled tokens you must label.\n"
        "\n"
        "Rules:\n"
        "  1. Use ONLY the entity types that appear in the support examples.\n"
        "  2. Label EVERY token in the query -- no skipping.\n"
        "  3. Use O for tokens that are not part of a named entity.\n"
        "  4. Output format: one token:::LABEL per line, nothing else.\n"
        "  5. Do NOT add explanations, headers, or blank lines between tokens."
    ),

    "P3": (
        "You are an expert named entity recognition (NER) system.\n"
        "\n"
        "You will receive support examples and a query sentence.\n"
        "\n"
        "Follow these steps:\n"
        "  Step 1 - Reason briefly: identify which query tokens are likely\n"
        "            named entities and which entity type they match from\n"
        "            the support examples. Write 1-3 sentences.\n"
        "  Step 2 - Output your labels. Write the exact line 'LABELS:' then\n"
        "            list every token as token:::LABEL, one per line.\n"
        "            Use O for non-entity tokens.\n"
        "            Use ONLY entity types from the support examples."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    tokens: list
    labels: list   # string labels, e.g. "person-actor", "O"


@dataclass
class Episode:
    n_way:        int
    k_shot:       int
    entity_types: list
    support:      list
    query:        list


@dataclass
class Prediction:
    tokens:       list
    gold_labels:  list
    pred_labels:  list
    raw_response: str
    latency_s:    float


@dataclass
class ConfigResult:
    n_way:           int
    k_range:         str
    k_shot:          int
    prompt_id:       str
    span_f1:         float
    precision:       float
    recall:          float
    total_latency_s: float
    n_episodes:      int
    predictions:     list


# ─────────────────────────────────────────────────────────────────────────────
# DATASET -- Few-NERD INTRA from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

def load_few_nerd_intra():
    """
    Download and parse Few-NERD INTRA test split from HuggingFace.
    Returns (sentences, label_names).
    """
    print("Downloading Few-NERD (INTRA) from HuggingFace...")
    print("This may take a moment on first run -- cached afterwards.\n")

    ds = load_dataset("DFKI-SLT/few-nerd", "intra", trust_remote_code=True)
    test = ds["test"]

    label_names = test.features["ner_tags"].feature.names

    sentences = []
    for row in test:
        tokens = row["tokens"]
        labels = [label_names[t] for t in row["ner_tags"]]
        if tokens:
            sentences.append(Sentence(tokens=tokens, labels=labels))

    print(f"Loaded {len(sentences)} sentences.")
    print(f"Label space ({len(label_names)} types): "
          f"{', '.join(label_names[:8])}...\n")
    
    return sentences, label_names


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def build_type_index(sentences):
    """Index sentences by the entity types they contain."""
    index = {}
    for sent in sentences:
        for label in sent.labels:
            if label != "O":
                index.setdefault(label, []).append(sent)
    return index


def sample_episode(type_index, n_way, k_shot, n_query=2):
    """
    Sample one N-way K-shot episode.
    Returns None if there is not enough data.
    """
    min_sents = k_shot + n_query
    eligible  = [t for t, sents in type_index.items() if len(sents) >= min_sents]

    if len(eligible) < n_way:
        return None

    chosen_types = random.sample(eligible, n_way)
    support_sents, query_sents = [], []

    for etype in chosen_types:
        pool = random.sample(type_index[etype], k_shot + n_query)
        support_sents.extend(pool[:k_shot])
        query_sents.extend(pool[k_shot:k_shot + n_query])

    random.shuffle(query_sents)

    return Episode(
        n_way=n_way,
        k_shot=k_shot,
        entity_types=chosen_types,
        support=support_sents,
        query=query_sents,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_support_block(episode):
    lines = [
        "=== SUPPORT EXAMPLES ===",
        f"Entity types in this episode: {', '.join(episode.entity_types)}",
        "",
    ]
    for i, sent in enumerate(episode.support):
        lines.append(f"[Example {i + 1}]")
        for token, label in zip(sent.tokens, sent.labels):
            lines.append(f"{token}:::{label}")
        lines.append("")
    return "\n".join(lines)


def build_query_block(sent):
    lines = ["=== QUERY ===", "Label every token below:", ""]
    lines.extend(sent.tokens)
    return "\n".join(lines)


def build_user_message(episode, query_sent):
    return build_support_block(episode) + "\n\n" + build_query_block(query_sent)


# ─────────────────────────────────────────────────────────────────────────────
# LM STUDIO API
# ─────────────────────────────────────────────────────────────────────────────

def call_model(system_prompt, user_message):
    """
    Call LM Studio. Returns (response_text, latency_seconds).
    Raises SystemExit with a helpful message if the server is unreachable.
    """
    payload = {
        "model": LM_STUDIO_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens":  512,
        "temperature": 0.0,   # greedy -- reproducibility matters
        "stream":      False,
    }

    t0 = time.time()
    try:
        r = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            "\n[ERROR] Cannot connect to LM Studio.\n"
            f"  Expected: {LM_STUDIO_URL}\n"
            "  -> Make sure LM Studio is open and a model is loaded.\n"
            "  -> If your port is not 1234, update LM_STUDIO_URL at the top of this file."
        )
    except Exception as e:
        text = f"PARSE_ERROR: {e}"

    return text, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_labels(response, tokens, prompt_id):
    """
    Extract predicted labels from model output, aligned to input tokens.
    Falls back to O for any position the model skipped or misformatted.
    """
    text = response

    # P3: only look at the section after LABELS:
    if prompt_id == "P3" and "LABELS:" in text:
        text = text.split("LABELS:", 1)[1]

    pairs = re.findall(r"(\S+):::(\S+)", text)

    pred_labels = ["O"] * len(tokens)
    for idx, (_, label) in enumerate(pairs):
        if idx < len(tokens):
            pred_labels[idx] = label

    return pred_labels


# ─────────────────────────────────────────────────────────────────────────────
# SPAN-LEVEL F1
# ─────────────────────────────────────────────────────────────────────────────

def extract_spans(labels):
    """
    Extract (start, end_exclusive, entity_type) from a label sequence.
    Handles both BIO tags and raw type labels (Few-NERD uses type-only).
    Consecutive identical non-O labels are merged into one span.
    """
    spans = set()
    i = 0
    while i < len(labels):
        lbl = labels[i]
        if lbl == "O":
            i += 1
            continue
        core  = lbl[2:] if lbl.startswith(("B-", "I-")) else lbl
        start = i
        i    += 1
        while i < len(labels):
            next_lbl  = labels[i]
            next_core = next_lbl[2:] if next_lbl.startswith(("B-", "I-")) else next_lbl
            if next_lbl == "O" or next_lbl.startswith("B-") or next_core != core:
                break
            i += 1
        spans.add((start, i, core))
    return spans


def compute_span_f1(predictions):
    """Micro-averaged span-level Precision, Recall, F1."""
    tp = fp = fn = 0
    for p in predictions:
        gold = extract_spans(p.gold_labels)
        pred = extract_spans(p.pred_labels)
        tp  += len(gold & pred)
        fp  += len(pred - gold)
        fn  += len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return round(f1, 4), round(prec, 4), round(rec, 4)


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_episodes(type_index, n_way, k_shot, prompt_id, n_episodes, label=""):
    """Run n_episodes and return all Prediction objects."""
    system_prompt = PROMPTS[prompt_id]
    all_preds     = []

    for ep_i in range(n_episodes):
        episode = sample_episode(type_index, n_way, k_shot)
        if episode is None:
            print(f"  [!] Episode {ep_i+1}: insufficient data, skipping")
            continue

        ep_preds = []
        for query_sent in episode.query:
            user_msg   = build_user_message(episode, query_sent)
            response, latency = call_model(system_prompt, user_msg)
            pred_labels = parse_labels(response, query_sent.tokens, prompt_id)

            ep_preds.append(Prediction(
                tokens=query_sent.tokens,
                gold_labels=query_sent.labels,
                pred_labels=pred_labels,
                raw_response=response,
                latency_s=round(latency, 3),
            ))

        all_preds.extend(ep_preds)
        ep_f1, _, _ = compute_span_f1(ep_preds)
        ep_lat = sum(p.latency_s for p in ep_preds)
        print(f"  {label} ep {ep_i+1:>2}/{n_episodes} | "
              f"F1={ep_f1:.3f} | {ep_lat:.1f}s | "
              f"types: {', '.join(episode.entity_types[:3])}...")

    return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# MINI EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_mini_experiment(type_index):
    """
    Compare P1 vs P2 vs P3 on MINI_CONFIG.
    Saves results/mini_experiment.json.
    Returns the winning prompt_id.
    """
    n_way   = MINI_CONFIG["n_way"]
    k_range = MINI_CONFIG["k_range"]
    k_min, k_max = K_RANGES[k_range]
    k_shot  = random.randint(k_min, k_max)

    print("=" * 60)
    print(f"MINI EXPERIMENT  {n_way}-way | {k_shot}-shot ({k_range})")
    print(f"Comparing P1, P2, P3 over {MINI_EPISODES} episodes each")
    print("=" * 60)

    mini_results = {}
    for pid in ["P1", "P2", "P3"]:
        print(f"\nPrompt {pid}:")
        preds = run_episodes(type_index, n_way, k_shot, pid,
                             MINI_EPISODES, label=f"[{pid}]")
        f1, prec, rec = compute_span_f1(preds)
        latency = sum(p.latency_s for p in preds)
        mini_results[pid] = {
            "span_f1":   f1,
            "precision": prec,
            "recall":    rec,
            "latency_s": round(latency, 2),
        }
        print(f"  -> {pid}: F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  "
              f"({latency:.1f}s total)")

    best = max(mini_results, key=lambda p: mini_results[p]["span_f1"])

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "config":      {"n_way": n_way, "k_shot": k_shot,
                        "k_range": k_range, "n_episodes": MINI_EPISODES},
        "results":     mini_results,
        "best_prompt": best,
        "prompts":     PROMPTS,
        "rationale": {
            "P1": "Minimal instruction -- baseline for format compliance",
            "P2": "Structured ICL with explicit support demonstration",
            "P3": "Chain-of-thought reasoning then structured output",
        },
    }
    path = RESULTS_DIR / "mini_experiment.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"\nBest prompt: {best} (F1={mini_results[best]['span_f1']:.4f})")
    print(f"Saved -> {path}\n")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(type_index, best_prompt):
    """Run all 8 INTRA configs with the winning prompt."""
    configs = [(n, k) for n in N_WAYS for k in K_RANGES]

    print("=" * 60)
    print(f"FULL EVALUATION  {len(configs)} configs  prompt: {best_prompt}")
    print("=" * 60)

    all_results = []
    for n_way, k_range in configs:
        k_min, k_max = K_RANGES[k_range]
        k_shot = random.randint(k_min, k_max)

        print(f"\n[{n_way}-way | {k_range} ({k_shot}-shot)]")
        preds = run_episodes(type_index, n_way, k_shot, best_prompt,
                             FULL_EPISODES, label=f"[{n_way}w-{k_range}]")

        f1, prec, rec = compute_span_f1(preds)
        latency = sum(p.latency_s for p in preds)

        all_results.append(ConfigResult(
            n_way=n_way,
            k_range=k_range,
            k_shot=k_shot,
            prompt_id=best_prompt,
            span_f1=f1,
            precision=prec,
            recall=rec,
            total_latency_s=round(latency, 2),
            n_episodes=FULL_EPISODES,
            predictions=[asdict(p) for p in preds],
        ))
        print(f"  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}  ({latency:.0f}s)")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE + PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results):
    """
    Save to results/instructllama_results.json.

    JSON structure:
    {
      "split":       "intra",
      "model":       "...",
      "best_prompt": "P2",
      "prompts":     { "P1": ..., "P2": ..., "P3": ... },
      "summary":     [ { n_way, k_range, k_shot, span_f1, precision,
                         recall, latency_s, n_episodes }, ... ],
      "full":        [ { ...summary..., predictions: [...] }, ... ]
    }

    Use "summary" for your results tables in the paper.
    Use "full.predictions" for qualitative error analysis.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = [
        {k: v for k, v in asdict(r).items() if k != "predictions"}
        for r in results
    ]

    payload = {
        "split":       "intra",
        "model":       LM_STUDIO_MODEL,
        "best_prompt": results[0].prompt_id if results else "N/A",
        "prompts":     PROMPTS,
        "summary":     summary,
        "full":        [asdict(r) for r in results],
    }

    path = RESULTS_DIR / "instructllama_results.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved -> {path}")

    # Console summary table
    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY  (INTRA | Llama 3.1 8B Instruct | ICL)")
    print("=" * 62)
    print(f"  {'N-way':<8} {'K-range':<10} {'K':<5} {'F1':<8} {'P':<8} {'R'}")
    print("  " + "-" * 56)
    for r in summary:
        print(f"  {r['n_way']:<8} {r['k_range']:<10} {r['k_shot']:<5} "
              f"{r['span_f1']:<8.4f} {r['precision']:<8.4f} {r['recall']:.4f}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  Llama 3.1 8B Instruct -- Few-NERD INTRA Evaluation")
    print("=" * 62 + "\n")

    sentences, _  = load_few_nerd_intra()
    type_index    = build_type_index(sentences)

    print(f"Entity types available: {len(type_index)}")
    print(f"Sample: {', '.join(sorted(type_index)[:8])}...\n")

    best_prompt = run_mini_experiment(type_index)
    results     = run_full_evaluation(type_index, best_prompt)
    save_results(results)

    print("\nDone.")


if __name__ == "__main__":
    main()