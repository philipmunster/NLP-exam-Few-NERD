import transformers
import torch

# AutoTokenizer and AutoModelForCausalLM not needed when using pipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM

# Use Instruct model - base model doesn't follow instructions well
# Change to "meta-llama/Llama-3.1-8B-Instruct" on HPC
model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"dtype": torch.float16},
    device_map="cpu"  # change to "auto" on HPC
)

messages = [
    {
        "role": "system",
        "content": "You are a Named Entity Recognition (NER) system. Tag each token as PERSON-Politician or O."
    },
    {
        "role": "user",
        "content": """Examples:
* Donald Trump visited the White House. --> [PERSON-Politician, O, O, O, O, O]
* Hillary Clinton flew to Paris. --> [PERSON-Politician, O, O, O, O]

Return ONLY a JSON object like: {"tokens": ["TOKEN1", "TOKEN2", ...], "labels": ["LABEL1", "LABEL2", ...]}

Sentence: Obama is visiting Washington."""
    }
]

result = pipeline(
    messages,
    max_new_tokens=800,
    do_sample=False,
    repetition_penalty=1.3
)

print(result[0]["generated_text"][-1]["content"])