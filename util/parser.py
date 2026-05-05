import ast
import json
import re
import pandas as pd

# creates colum names for the dataframe 
COLUMNS = ["prompt_type", "N", "K", "col4", "seed", "prompt_answer", "gold_standard"]

# Regex for a valid NER label (e.g. "O", "event-election", "person-actor")
LABEL_RE = re.compile(r'^(?:O|[a-z][\w]*(?:-[\w]+)+)$', re.IGNORECASE)


def _parse_json_arrays(text):
    """
    Collects complete JSON arrays from llama output
    Returns a flat list of dicts if successful, or None if no valid arrays found.
    """
    results = []
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']' and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        results.extend(parsed)
                except json.JSONDecodeError:
                    pass
                start = None

    return results or None



def _parse_colon_format(text: str) -> list | None:
    """
    Collects token:::LABEL or token ::: LABEL, one per line.
    Returns a flat list of dicts if successful, or None if no valid lines found.
    """

    # Restricts to only look after "Answer:" to avoid passing extra information
    m = re.search(r'(?i)^answer\s*:', text, re.MULTILINE)
    if m:
        text = text[m.end():]

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'\s*:::\s*', line, maxsplit=1)
        if len(parts) != 2:
            continue
        token_part, label = parts[0], parts[1].strip()
        # Reject if label has spaces or doesn't look like a NER label
        if not label or not LABEL_RE.match(label):
            continue
        # Strip leading numbering: "1. ", "1 ", "N. "
        token = re.sub(r'^\d+[\.\s]+', '', token_part).strip()
        if token:
            records.append({"token": token, "label": label})

    return records or None



def extract_json_ner(raw: str) -> list | None:
    """
   Takes as an input the answer string from the LLM output and tries to extract a list of {"token": ..., "label": ...} dicts.

    """
    if not isinstance(raw, str):
        return None


    try:
        outer = ast.literal_eval(raw)
        inner = outer[0] if isinstance(outer, list) and outer else raw
    except Exception:
        inner = raw

    result = _parse_json_arrays(inner)
    if result:
        return result
    
    result = _parse_colon_format(inner)
    if result:
        return result

    return None


def parse_ner_column(df, answer_col="prompt_answer", out_col="json_format"):
    """
    Add a `json_format` column to *df* by applying extract_json_ner() to
    every row on the dataset.
    """
    df = df.copy()
    df[out_col] = df[answer_col].apply(extract_json_ner)
    return df



if __name__ == "__main__":
    import sys

    path    = "ner_results.csv"
    out_path = "ner_results_with_json.csv"

    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = parse_ner_column(df, answer_col="prompt_answer", out_col="json_format")

    total      = len(df)
    parsed_ok  = df["json_format"].notna().sum()
    parse_fail = total - parsed_ok

    print(f"Rows processed : {total}")
    print(f"  parsed OK    : {parsed_ok}  ({parsed_ok / total:.1%})")
    print(f"  parse failed : {parse_fail} ({parse_fail / total:.1%})")
    if parse_fail:
        print(f"  failed rows  : {list(df[df['json_format'].isna()].index)}")

    print()
    for i in range(min(3, total)):
        result = df["json_format"].iloc[i]
        n = len(result) if isinstance(result, list) else "N/A"
        preview = result[:3] if isinstance(result, list) else result
        print(f"Row {i}: {n} tokens  |  first 3 -> {preview}")

    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")