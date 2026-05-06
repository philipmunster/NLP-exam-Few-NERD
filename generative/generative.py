import transformers
import torch
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.data_loader_generative import data_loader_generative
from transformers import BitsAndBytesConfig

# Change to "meta-llama/Llama-3.1-8B-Instruct" on HPC
model_id = "meta-llama/Llama-3.1-8B-Instruct"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map="auto"
)

n_list = [5, 10]
k_list = [1, 5]
seed_list = range(1, 501)
test_size = 5

for n in n_list:
    for k in k_list:
        for seed in seed_list:

            print(f"Iteration using n = {n}, k = {k}, seed = {seed}")

            classes, support_df, test_df = data_loader_generative(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'inter', 'test.txt'),
                n, k, test_size, seed
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

            prompt = f"""You are given a few-shot named entity recognition task.
                    You will receive:
                    - SUPPORT EXAMPLES: sentences where every token is already labeled.
                    - A QUERY sentence: unlabeled tokens you must label.
                    Your task is to infer the labeling pattern from the SUPPORT EXAMPLES and apply it to the QUERY sentence.
                    Rules:
                    1. Use ONLY the entity types that appear in the support examples.
                    2. Label EVERY token in the query -- no skipping.
                    3. Use O for tokens that are not part of a named entity.
                    4. Output format: one token:::LABEL per line, nothing else.
                    5. Do NOT add explanations, headers, or blank lines between tokens.
                    SUPPORT EXAMPLES:
                    {examples_str}
                    QUERY SENTENCE:
                    {test_str}
                    OUTPUT:"""

            output = pipeline(prompt, max_new_tokens=2048)

            #raw = output[0]["generated_text"][-1]["content"]
            raw = output[0]["generated_text"][len(prompt):]

            #safe outputs in a df [n, k, test_size, seed, model_output, ground_truth]
            # Ground truth from test_df
            ground_truth = test_df[["sentence", "labels"]].to_dict(orient="records")


            row = {
                "n": n,
                "k": k,
                "seed": seed,
                "model_output": [raw],   # wrap in list so it stores as object
                "ground_truth": [ground_truth],
            }

            results_df = pd.DataFrame([row])

            output_path = "generative/output/ner_results_final.csv"
            results_df.to_csv(
                output_path,
                mode='a',
                index=False,
                header=not os.path.exists(output_path),
            )