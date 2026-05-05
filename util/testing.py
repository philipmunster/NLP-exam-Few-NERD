import ast
import sys
import pandas as pd
from collections import defaultdict
 
 
# =============================================================================
# Step 1 — Alignment: find the gold sentence that matches the model output
# =============================================================================
 
def find_matching_gold_sentence(pred_tokens: list[str], gold_sentences: list[dict]) -> int:
    """
    The gold standard contains all sentences (support examples + query).
    The model only labels the query sentence.
    We find the gold sentence that best matches the model output by counting
    how many tokens at the start of both sequences agree (prefix match).
 
    Returns the index of the best-matching sentence in gold_sentences.
    """
    best_idx, best_score = 0, -1
    for j, sent in enumerate(gold_sentences):
        gold_tokens = sent["sentence"].split()
        # Count matching tokens from the start (prefix overlap)
        score = sum(1 for p, g in zip(pred_tokens, gold_tokens) if p == g)
        if score > best_score:
            best_score = score
            best_idx = j
    return best_idx
 
 
def align_prediction_to_gold(jf: list[dict], gold_sentences: list[dict]) -> tuple[list, list]:
    """
    Given a list of predicted {"token", "label"} dicts and the full list of
    gold sentences, returns two aligned lists:
        pred_labels : labels predicted by the model
        true_labels : corresponding gold labels
 
    Alignment is truncated to the shorter of the two sequences.
    """
    pred_tokens = [d["token"] for d in jf]
    best_idx    = find_matching_gold_sentence(pred_tokens, gold_sentences)
 
    gold_sent   = gold_sentences[best_idx]
    gold_labels = gold_sent["labels"]
 
    n           = min(len(jf), len(gold_labels))
    pred_labels = [d["label"] for d in jf[:n]]
    true_labels = gold_labels[:n]
 
    return pred_labels, true_labels
 
 
# =============================================================================
# Step 2 — Token-level metrics for one row
# =============================================================================
 
def row_metrics(pred_labels: list[str], true_labels: list[str]) -> dict:
    """
    Compute token-level metrics for a single row.
 
    Returns a dict with:
        accuracy        overall token accuracy (including O tokens)
        macro_f1        unweighted average F1 across all non-O labels present
        weighted_f1     support-weighted average F1
        per_label       dict of {label: {precision, recall, f1, support}}
        n_tokens        number of aligned tokens
        n_pred_entities number of tokens the model predicted as non-O
        n_true_entities number of tokens that are truly non-O
    """
    assert len(pred_labels) == len(true_labels), "Lists must be same length"
 
    # Collect all non-O labels that appear in either prediction or gold
    all_entity_labels = set(true_labels) | set(pred_labels)
    all_entity_labels.discard("O")
 
    per_label = {}
    for lbl in all_entity_labels:
        tp = sum(1 for p, t in zip(pred_labels, true_labels) if p == lbl and t == lbl)
        fp = sum(1 for p, t in zip(pred_labels, true_labels) if p == lbl and t != lbl)
        fn = sum(1 for p, t in zip(pred_labels, true_labels) if p != lbl and t == lbl)
 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = tp + fn  # total true positives for this label in gold
 
        per_label[lbl] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   support,
        }
 
    # Macro F1: simple average across labels (ignores class imbalance)
    macro_f1 = (sum(v["f1"] for v in per_label.values()) / len(per_label)
                if per_label else 0.0)
 
    # Weighted F1: each label's F1 weighted by how often it appears in gold
    total_support = sum(v["support"] for v in per_label.values())
    weighted_f1   = (sum(v["f1"] * v["support"] for v in per_label.values()) / total_support
                     if total_support > 0 else 0.0)
 
    # Token accuracy (including O — useful for overall correctness)
    accuracy = (sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
                / len(true_labels))
 
    return {
        "accuracy":         accuracy,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "per_label":        per_label,
        "n_tokens":         len(true_labels),
        "n_pred_entities":  sum(1 for p in pred_labels if p != "O"),
        "n_true_entities":  sum(1 for t in true_labels if t != "O"),
    }
 
 
# =============================================================================
# Step 3 — Aggregate metrics across all rows
# =============================================================================
 
def aggregate_metrics(rows: list[dict]) -> dict:
    """
    Given a list of row_metrics dicts, compute corpus-level aggregates:
        mean_accuracy, mean_macro_f1, mean_weighted_f1
        micro_f1      (pool all TPs/FPs/FNs across rows and labels)
        per_label     pooled per-label stats across all rows
    """
    if not rows:
        return {}
 
    # Simple means
    mean_accuracy    = sum(r["accuracy"]    for r in rows) / len(rows)
    mean_macro_f1    = sum(r["macro_f1"]    for r in rows) / len(rows)
    mean_weighted_f1 = sum(r["weighted_f1"] for r in rows) / len(rows)
 
    # Pool per-label counts for micro metrics
    pooled = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        for lbl, stats in r["per_label"].items():
            # Reconstruct tp/fp/fn from precision/recall/support
            support = stats["support"]          # = tp + fn
            prec    = stats["precision"]
            rec     = stats["recall"]
            # tp = recall * support
            tp = round(rec * support)
            fn = support - tp
            fp = round(tp / prec - tp) if prec > 0 else 0
            pooled[lbl]["tp"] += tp
            pooled[lbl]["fp"] += fp
            pooled[lbl]["fn"] += fn
 
    per_label_agg = {}
    total_tp = total_fp = total_fn = 0
    for lbl, counts in pooled.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label_agg[lbl] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": tp + fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
 
    # Micro F1: computed from pooled TP/FP/FN (treats every token equally)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0.0)
 
    return {
        "n_rows":           len(rows),
        "mean_accuracy":    mean_accuracy,
        "mean_macro_f1":    mean_macro_f1,
        "mean_weighted_f1": mean_weighted_f1,
        "micro_f1":         micro_f1,
        "per_label":        per_label_agg,
    }
 
 
# =============================================================================
# Step 4 — Main function: run everything
# =============================================================================
 
def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main evaluation function.
 
    For each row in df that has a non-null json_format:
      1. Parse both json_format and gold_standard
      2. Align prediction to the matching gold sentence
      3. Compute row-level metrics
 
    Returns df with new columns added:
        accuracy, macro_f1, weighted_f1, n_tokens,
        n_pred_entities, n_true_entities, skip_reason
    """
    results = []
 
    for i, row in df.iterrows():
        # --- Skip rows where the model failed to produce valid NER ---
        if pd.isna(row["json_format"]):
            results.append({
                "accuracy": None, "macro_f1": None, "weighted_f1": None,
                "n_tokens": None, "n_pred_entities": None, "n_true_entities": None,
                "skip_reason": "parse_failed",
            })
            continue
 
        try:
            jf             = ast.literal_eval(row["json_format"])
            gold_sentences = ast.literal_eval(row["gold_standard"])[0]
 
            pred_labels, true_labels = align_prediction_to_gold(jf, gold_sentences)
            metrics = row_metrics(pred_labels, true_labels)
 
            results.append({
                "accuracy":         round(metrics["accuracy"],    4),
                "macro_f1":         round(metrics["macro_f1"],    4),
                "weighted_f1":      round(metrics["weighted_f1"], 4),
                "n_tokens":         metrics["n_tokens"],
                "n_pred_entities":  metrics["n_pred_entities"],
                "n_true_entities":  metrics["n_true_entities"],
                "skip_reason":      None,
            })
 
        except Exception as e:
            results.append({
                "accuracy": None, "macro_f1": None, "weighted_f1": None,
                "n_tokens": None, "n_pred_entities": None, "n_true_entities": None,
                "skip_reason": f"error: {e}",
            })
 
    metrics_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, metrics_df], axis=1)
 
 
# =============================================================================
# Printing helpers
# =============================================================================
 
def print_aggregate(title: str, agg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}  (n={agg['n_rows']} rows)")
    print(f"{'='*60}")
    print(f"  Token Accuracy    : {agg['mean_accuracy']:.3f}")
    print(f"  Mean Macro F1     : {agg['mean_macro_f1']:.3f}")
    print(f"  Mean Weighted F1  : {agg['mean_weighted_f1']:.3f}")
    print(f"  Micro F1          : {agg['micro_f1']:.3f}")
    print()
    print(f"  {'Label':<35} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for lbl, s in sorted(agg["per_label"].items(), key=lambda x: -x[1]["support"]):
        print(f"  {lbl:<35} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f} {s['support']:>8}")
 
 
# =============================================================================
# CLI entry point
# =============================================================================
 
if __name__ == "__main__":
    path     = sys.argv[1] if len(sys.argv) > 1 else "ner_results_with_json.csv"
    out_path = path.replace(".csv", "_metrics.csv")
 
    # --- Load and evaluate ---
    df     = pd.read_csv(path)
    df_out = evaluate(df)
 
    # --- Row-level summary ---
    evaluated = df_out[df_out["skip_reason"].isna()]
    skipped   = df_out[df_out["skip_reason"].notna()]
    print(f"\nTotal rows     : {len(df_out)}")
    print(f"Evaluated      : {len(evaluated)}")
    print(f"Skipped (None) : {len(skipped)}")
 
    # --- Overall aggregate ---
    # Rebuild row_metrics dicts for aggregation
    all_row_metrics = []
    for i, row in evaluated.iterrows():
        jf             = ast.literal_eval(row["json_format"])
        gold_sentences = ast.literal_eval(row["gold_standard"])[0]
        pred_labels, true_labels = align_prediction_to_gold(jf, gold_sentences)
        all_row_metrics.append(row_metrics(pred_labels, true_labels))
 
    overall_agg = aggregate_metrics(all_row_metrics)
    print_aggregate("OVERALL", overall_agg)
 
    # --- By prompt_type ---
    for pt, group in df_out.groupby("prompt_type"):
        group_eval = group[group["skip_reason"].isna()]
        if group_eval.empty:
            continue
        group_metrics = []
        for i, row in group_eval.iterrows():
            jf             = ast.literal_eval(row["json_format"])
            gold_sentences = ast.literal_eval(row["gold_standard"])[0]
            pred_labels, true_labels = align_prediction_to_gold(jf, gold_sentences)
            group_metrics.append(row_metrics(pred_labels, true_labels))
        agg = aggregate_metrics(group_metrics)
        print_aggregate(pt, agg)
 
    # --- Save enriched CSV ---
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")