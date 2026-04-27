import transformers
import torch
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.data_loader_generative import data_loader_generative

n = 1
k = 1
test_size = 10
seed = 21

classes, support_df, test_df = data_loader_generative(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'inter', 'test.txt'),
    n, k, test_size, seed
)

# import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# print(classes)
# print("Support DF:")
# print(support_df)
# print("Test DF:")
# print(test_df)





# Change to "meta-llama/Llama-3.1-8B-Instruct" on HPC
model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.float16},
    device_map="cpu"  # change to "auto" on HPC
)

# Build class list string
class_str = ", ".join(classes)

# Build few-shot examples from support_df
example_lines = []
for _, row in support_df.iterrows():
    tokens = row["sentence"].split()
    labels = row["labels"]
    example_lines.append(f"* {row['sentence']} --> [{', '.join(labels)}]")
examples_str = "\n".join(example_lines)

# Build test sentences from test_df
test_lines = []
for _, row in test_df.iterrows():
    test_lines.append(f"Sentence: {row['sentence']}")
test_str = "\n".join(test_lines)

messages = [
    {
        "role": "system",
        "content": f"You are a Named Entity Recognition (NER) system. Tag each token as {class_str} or O."
    },
    {
        "role": "user",
        "content": f"""Examples:
{examples_str}

Return ONLY a JSON object like: {{"sentence": ["TOKEN1", "TOKEN2", ...], "labels": ["LABEL1", "LABEL2", ...]}}

{test_str}"""
    }
]

output = pipeline(messages, max_new_tokens=512)

raw = output[0]["generated_text"][-1]["content"]

#safe outputs in a df [n, k, test_size, seed, model_output, ground_truth]
# Ground truth from test_df
ground_truth = test_df[["sentence", "labels"]].to_dict(orient="records")


row = {
    "n": n,
    "k": k,
    "test_size": test_size,
    "seed": seed,
    "model_output": [raw],   # wrap in list so it stores as object
    "ground_truth": [ground_truth],
}

results_df = pd.DataFrame([row])

results_df.to_csv("generative/output/ner_results.csv", index=False)