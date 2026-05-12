import pandas as pd
from collections import defaultdict
from prompt_testing import (
    extract_spans,
    align_predictions_to_gold,
    compute_prf,
)
import ast


def coarse(label):
    return label.split("-")[0] if "-" in label else label


def compute_errors(gold_spans, pred_spans):
    """
    Compute within_error and outer_error fractions for one episode,
    matching the Few-NERD paper's error analysis (Table 7).

    within_error: of all FP spans, what fraction were a type confusion
                  within the same coarse group
                  e.g. predicted person-actor when gold was person-scholar

    outer_error:  of all FP spans, what fraction were a type confusion
                  across coarse groups
                  e.g. predicted location-gpe when gold was person-actor

    Both are expressed as fractions of total FP count.
    A FP span can only be within OR outer, not both.
    A FP span that has no position overlap with any gold span is neither
    (it is a pure hallucination) — counted in neither error type.
    """
    fp_spans = pred_spans - gold_spans

    if not fp_spans:
        return 0.0, 0.0

    gold_pos = defaultdict(set)
    for start, end, lbl in gold_spans:
        gold_pos[(start, end)].add(lbl)

    within = 0
    outer  = 0

    for start, end, pred_lbl in fp_spans:
        gold_types_at_pos = gold_pos.get((start, end), set())
        if not gold_types_at_pos:
            continue 

        pred_coarse  = coarse(pred_lbl)
        gold_coarses = {coarse(g) for g in gold_types_at_pos}

        if pred_coarse in gold_coarses:
            within += 1   # right position, right coarse, wrong fine type
        else:
            outer  += 1   # right position, wrong coarse type entirely

    total_fp = len(fp_spans)
    return within / total_fp, outer / total_fp


def build_proto_rows(evaluated_rows, df):
    # Accumulators keyed by fine-grained type label
    fine = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0,
                                "within": 0.0, "outer": 0.0,
                                "support": 0, "query_cnt": 0,
                                "fp_count_for_error": 0})

    total_tp = total_fp = total_fn = 0

    for result in evaluated_rows:
        i = result["row"]
        row = df.iloc[i]

        # Re-parse to get raw spans for error analysis
        try:
            pred_items     = ast.literal_eval(row["json_format"])
            gold_sentences = ast.literal_eval(row["ground_truth"])[0]
            pred_labels, gold_labels = align_predictions_to_gold(
                pred_items, gold_sentences
            )
            gold_spans = extract_spans(gold_labels)
            pred_spans = extract_spans(pred_labels)
        except Exception:
            continue

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        # Per fine-grained type accumulation
        for span in gold_spans & pred_spans:
            fine[span[2]]["tp"]        += 1
            fine[span[2]]["support"]   += 1
            fine[span[2]]["query_cnt"] += 1

        for span in pred_spans - gold_spans:
            fine[span[2]]["fp"]        += 1
            fine[span[2]]["query_cnt"] += 1
            fine[span[2]]["fp_count_for_error"] += 1

        for span in gold_spans - pred_spans:
            fine[span[2]]["fn"]      += 1
            fine[span[2]]["support"] += 1

        # Within/outer errors — computed per episode then accumulated
        w, o = compute_errors(gold_spans, pred_spans)
        fp_count = len(pred_spans - gold_spans)
        for lbl in {s[2] for s in pred_spans - gold_spans}:
            # Distribute episode-level error fractions proportionally
            fine[lbl]["within"] += w
            fine[lbl]["outer"]  += o

    # ── Build output rows ────────────────────────────────────────────────────

    rows = []

    p, r, f1 = compute_prf(total_tp, total_fp, total_fn)
    rows.append({
        "type":         "overall",
        "precision":    p,
        "recall":       r,
        "f1":           f1,
        "fp_cnt":       total_fp,
        "fn_cnt":       total_fn,
        "within_error": 0.0,
        "outer_error":  0.0,
        "support":      "",      
        "query_cnt":    ";",   
    })

    coarse_agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0,
                                       "within": 0.0, "outer": 0.0,
                                       "support": 0, "query_cnt": 0})
    for lbl, stats in fine.items():
        c = coarse(lbl)
        coarse_agg[c]["tp"]        += stats["tp"]
        coarse_agg[c]["fp"]        += stats["fp"]
        coarse_agg[c]["fn"]        += stats["fn"]
        coarse_agg[c]["within"]    += stats["within"]
        coarse_agg[c]["outer"]     += stats["outer"]
        coarse_agg[c]["support"]   += stats["support"]
        coarse_agg[c]["query_cnt"] += stats["query_cnt"]

    for c, s in sorted(coarse_agg.items()):
        p, r, f1 = compute_prf(s["tp"], s["fp"], s["fn"])
        fp = s["fp"]
        rows.append({
            "type":         c,
            "precision":    p,
            "recall":       r,
            "f1":           f1,
            "fp_cnt":       fp,
            "fn_cnt":       s["fn"],
            "within_error": s["within"] / fp if fp > 0 else 0.0,
            "outer_error":  s["outer"]  / fp if fp > 0 else 0.0,
            "support":      s["support"],
            "query_cnt":    s["query_cnt"],
        })

    for lbl, s in sorted(fine.items()):
        p, r, f1 = compute_prf(s["tp"], s["fp"], s["fn"])
        fp = s["fp"]
        rows.append({
            "type":         lbl,
            "precision":    p,
            "recall":       r,
            "f1":           f1,
            "fp_cnt":       fp,
            "fn_cnt":       s["fn"],
            "within_error": s["within"] / fp if fp > 0 else 0.0,
            "outer_error":  s["outer"]  / fp if fp > 0 else 0.0,
            "support":      s["support"],
            "query_cnt":    s["query_cnt"],
        })

    return rows

def results_to_proto_csv(evaluated_rows, df, out_path):
    rows = build_proto_rows(evaluated_rows, df)
    out  = pd.DataFrame(rows, columns=[
        "type", "precision", "recall", "f1",
        "fp_cnt", "fn_cnt", "within_error", "outer_error",
        "support", "query_cnt"
    ])
    out.to_csv(out_path, index=False)
    print(f"Saved proto-format CSV -> {out_path}")
    return out



if __name__ == "__main__":
    import sys
    import os
    from prompt_testing import evaluate

    path    = "ner_results_with_json.csv"
    out_dir = path.replace(".csv", "_proto_format")

    os.makedirs(out_dir, exist_ok=True)

    df     = pd.read_csv(path)
    groups = df.groupby(["n", "k"])

    print(f"Loaded {len(df)} rows | {len(groups)} (n, k) groups")

    for (n, k), group_df in groups:
        group_df    = group_df.reset_index(drop=True)
        all_results = evaluate(group_df)
        evaluated   = [r for r in all_results if not r["skipped"]]

        out_path = os.path.join(out_dir, f"proto-n{n}-k{k}.csv")
        results_to_proto_csv(evaluated, group_df, out_path)

    print(f"\nDone. {len(groups)} files saved to: {out_dir}/")
