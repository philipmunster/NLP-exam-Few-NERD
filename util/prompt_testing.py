import ast
import sys
import json
from collections import defaultdict
import pandas as pd


def extract_spans(labels):
    spans = set()
    i = 0
    while i < len(labels): 
        label = labels[i]

        if label == "O":
            i += 1
            continue

        start = i
        i += 1

        # Extend span while the same type continues
        while i < len(labels) and labels[i] == label:
            i += 1

        spans.add((start, i, label))

    return spans

def find_sequence(pred_tokens, gold_tokens):
    if not gold_tokens:
        return -1

    anchor    = gold_tokens[0]
    match_len = min(3, len(gold_tokens))

    for i, tok in enumerate(pred_tokens):
        if tok == anchor:
            if pred_tokens[i: i + match_len] == gold_tokens[:match_len]:
                return i

    return -1


def _strip_bio(label):
    """Normalize BIO/BIOES prefixes and casing to plain Few-NERD labels."""
    if len(label) > 2 and label[0] in "BIES" and label[1] == "-":
        label = label[2:]
    label = label.lower()
    return "O" if label == "o" else label


def align_predictions_to_gold(pred_items, gold_sentences):

    pred_tokens = [d["token"] for d in pred_items]
    pred_labels = [_strip_bio(d["label"]) for d in pred_items]

    all_pred = []
    all_gold = []

    for sent in gold_sentences:
        gold_tokens = sent["sentence"].split()
        gold_lbls   = sent["labels"]
        n_gold      = len(gold_tokens)

        start = find_sequence(pred_tokens, gold_tokens)

        if start == -1:
            # Sentence not found in predictions — all its entities are FN
            all_pred.extend(["O"] * n_gold)
            all_gold.extend(gold_lbls)
        else:
            # Extract the predicted labels for this sentence's token range
            n = min(n_gold, len(pred_tokens) - start)
            all_pred.extend(pred_labels[start: start + n])
            all_gold.extend(gold_lbls[:n])

            # If model produced fewer tokens than gold, pad the rest with O
            if n < n_gold:
                all_pred.extend(["O"] * (n_gold - n))
                all_gold.extend(gold_lbls[n:])

    return all_pred, all_gold


def span_counts(pred_labels, gold_labels):

    gold_spans = extract_spans(gold_labels)
    pred_spans = extract_spans(pred_labels)

    tp = len(gold_spans & pred_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)

    return tp, fp, fn


def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)

def aggregate(rows):

    total_tp = total_fp = total_fn = 0
    per_type = defaultdict(lambda: [0, 0, 0])  # [tp, fp, fn]

    for row in rows:
        total_tp += row["tp"]
        total_fp += row["fp"]
        total_fn += row["fn"]
        for lbl, (tp, fp, fn) in row["per_type"].items():
            per_type[lbl][0] += tp
            per_type[lbl][1] += fp
            per_type[lbl][2] += fn

    p, r, f1 = compute_prf(total_tp, total_fp, total_fn)

    # Per-type breakdown — exclude zero-support labels from macro F1
    per_type_metrics = {}
    macro_f1_values  = []
    for lbl, (tp, fp, fn) in per_type.items():
        lp, lr, lf = compute_prf(tp, fp, fn)
        support = tp + fn
        per_type_metrics[lbl] = {
            "precision": lp, "recall": lr, "f1": lf, "support": support
        }
        if support > 0:
            macro_f1_values.append(lf)

    macro_f1 = (sum(macro_f1_values) / len(macro_f1_values)
                if macro_f1_values else 0.0)

    return {
        "n_rows":   len(rows),
        "micro_f1": f1,
        "precision": p,
        "recall":   r,
        "macro_f1": round(macro_f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_type": per_type_metrics,
    }


def evaluate(df):
    results = []

    for i, row in df.iterrows():

        if pd.isna(row["json_format"]):
            results.append({
                "row":         i,
                "skipped":     True,
                "skip_reason": "json_format is null — model parse failed",
            })
            continue

        try:
            pred_items     = ast.literal_eval(row["json_format"])
            gold_sentences = ast.literal_eval(row["ground_truth"])[0]

            pred_labels, gold_labels = align_predictions_to_gold(
                pred_items, gold_sentences
            )

            tp, fp, fn = span_counts(pred_labels, gold_labels)

            # Per-type TP/FP/FN for this row
            gold_spans   = extract_spans(gold_labels)
            pred_spans   = extract_spans(pred_labels)
            row_per_type = defaultdict(lambda: [0, 0, 0])
            for span in gold_spans & pred_spans:
                row_per_type[span[2]][0] += 1   # tp
            for span in pred_spans - gold_spans:
                row_per_type[span[2]][1] += 1   # fp
            for span in gold_spans - pred_spans:
                row_per_type[span[2]][2] += 1   # fn

            results.append({
                "row":          i,
                "skipped":      False,
                "tp":           tp,
                "fp":           fp,
                "fn":           fn,
                "per_type":     {k: tuple(v) for k, v in row_per_type.items()},
                "n_gold_spans": len(gold_spans),
                "n_pred_spans": len(pred_spans),
            })

        except Exception as e:
            results.append({
                "row":         i,
                "skipped":     True,
                "skip_reason": f"error: {e}",
            })

    return results


def print_results(title, agg, n_total, n_skipped):
    skip_rate = n_skipped / n_total if n_total > 0 else 0.0

    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"  Rows: {agg['n_rows']} evaluated, {n_skipped} skipped "
          f"({skip_rate:.0%} skip rate)")
    print(f"{'='*62}")
    print(f"  Micro F1    : {agg['micro_f1']:.4f}")
    print(f"  Precision   : {agg['precision']:.4f}")
    print(f"  Recall      : {agg['recall']:.4f}")
    print(f"  Macro F1    : {agg['macro_f1']:.4f}  "
          f"(over labels with support > 0 only)")
    print(f"  TP / FP / FN: {agg['total_tp']} / "
          f"{agg['total_fp']} / {agg['total_fn']}")

    if agg["per_type"]:
        print()
        print(f"  {'Label':<35} {'F1':>6} {'P':>6} {'R':>6} {'Support':>8}")
        print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
        for lbl, s in sorted(
            agg["per_type"].items(),
            key=lambda x: -x[1]["support"]
        ):
            if s["support"] == 0:
                continue
            print(f"  {lbl:<35} {s['f1']:>6.3f} {s['precision']:>6.3f} "
                  f"{s['recall']:>6.3f} {s['support']:>8}")


if __name__ == "__main__":
    path     = sys.argv[1] if len(sys.argv) > 1 else "ner_results_with_json.csv"
    out_path = path.replace(".csv", "_metrics.json")

    df = pd.read_csv(path)
    print(f"\nLoaded {len(df)} rows from {path}")

    all_results = evaluate(df)

    evaluated = [r for r in all_results if not r["skipped"]]
    skipped   = [r for r in all_results if r["skipped"]]

    print(f"Evaluated: {len(evaluated)}  |  Skipped: {len(skipped)}")

    # Overall
    overall_agg = aggregate(evaluated)
    print_results("OVERALL — Span-level Micro F1 (Few-NERD protocol)",
                  overall_agg, len(df), len(skipped))

    # Per prompt_type
    for pt in df["prompt_type"].unique():
        pt_indices = set(df[df["prompt_type"] == pt].index)
        pt_results = [r for r in evaluated if r["row"] in pt_indices]
        pt_skipped = [r for r in skipped   if r["row"] in pt_indices]

        if not pt_results:
            continue

        pt_agg = aggregate(pt_results)
        print_results(pt, pt_agg,
                      n_total=len(pt_indices),
                      n_skipped=len(pt_skipped))

    # Save to JSON
    output = {
        "schema":    "IO — matching Few-NERD paper (Ding et al., ACL 2021)",
        "averaging": "micro — TP/FP/FN pooled before computing P/R/F1",
        "overall":   overall_agg,
        "per_prompt_type": {},
    }
    for pt in df["prompt_type"].unique():
        pt_indices = set(df[df["prompt_type"] == pt].index)
        pt_results = [r for r in evaluated if r["row"] in pt_indices]
        if pt_results:
            output["per_prompt_type"][pt] = aggregate(pt_results)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")