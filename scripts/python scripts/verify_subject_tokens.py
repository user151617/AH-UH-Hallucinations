import torch
from transformers import AutoTokenizer
import pandas as pd

MODEL_NAME = "EleutherAI/pythia-6.9b"
DATA_PATH = "/workspace/project/results/pythia-6.9b_with_jsdiv.parquet"
N_SAMPLES = 10

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def find_subject_token_positions(prompt, subject, tokenizer):
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding["offset_mapping"][0].tolist()
    char_start = prompt.find(subject)
    if char_start == -1:
        return [], encoding
    char_end = char_start + len(subject)
    positions = [
        idx for idx, (s, e) in enumerate(offsets)
        if s != e and s < char_end and e > char_start
    ]
    return positions, encoding

df = pd.read_parquet(DATA_PATH)

for i in range(N_SAMPLES):
    row = df.iloc[i]
    prompt = row['question']
    subject = row['subj']
    subj_positions, encoding = find_subject_token_positions(prompt, subject, tokenizer)
    input_ids = encoding['input_ids']
    decoded = tokenizer.convert_ids_to_tokens(input_ids[0][subj_positions]) if subj_positions else []
    print(f"Subject: {subject!r:30} -> Positions: {subj_positions} -> Tokens: {decoded}")
