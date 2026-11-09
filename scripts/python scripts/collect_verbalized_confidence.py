import torch
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--input_path", required=True, help="the *_with_features.parquet file")
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, torch_dtype=torch.float16, device_map="auto",
)
model.eval()

df = pd.read_parquet(args.input_path)

def build_confidence_prompt(question, generation):
   
    return (
        f"Question: {question}\n"
        f"Answer: {generation.strip()}\n"
        f"On a scale of 0 to 100, how confident are you that this answer is correct? "
        f"Respond with only a number.\nConfidence:"
    )

def extract_confidence_score(text):
    match = re.search(r'\d+', text)
    if match:
        val = int(match.group())
        return min(val, 100) / 100.0  
    return None

BATCH_SIZE = 16
confidences = []
prompts = [build_confidence_prompt(row['question'], row['generation']) for _, row in df.iterrows()]

for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch = prompts[i:i+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    for j, output in enumerate(outputs):
        input_len = inputs['input_ids'][j].shape[0]
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        confidences.append(extract_confidence_score(text))

df['verbalized_confidence'] = confidences
n_missing = sum(c is None for c in confidences)
print(f"Could not parse a number for {n_missing} / {len(df)} examples")
df.to_parquet(args.output_path)
print(f"Saved to {args.output_path}")
