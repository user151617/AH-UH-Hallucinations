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
parser.add_argument("--max_new_tokens", type=int, default=150)
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

def build_cot_prompt(question, generation):
    return (
        f"Question: {question}\n"
        f"Proposed answer: {generation.strip()}\n\n"
        f"Let's think step by step about whether this answer is correct. "
        f"Consider what you know about the subject, then reason through it. "
        f"End your response with 'Confidence: X' where X is 0-100.\n"
        f"Reasoning:"
    )

def extract_final_confidence(text):
    match = re.search(r'[Cc]onfidence[:\s]+(\d+)', text)
    if match:
        val = int(match.group(1))
        return min(val, 100) / 100.0
    # Fallback: take the LAST number mentioned anywhere in the reasoning
    numbers = re.findall(r'\b(\d{1,3})\b', text)
    if numbers:
        val = int(numbers[-1])
        if val <= 100:
            return val / 100.0
    return None

BATCH_SIZE = 8  
cot_texts = []
cot_confidences = []
prompts = [build_cot_prompt(row['question'], row['generation']) for _, row in df.iterrows()]

for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch = prompts[i:i+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    for j, output in enumerate(outputs):
        input_len = inputs['input_ids'][j].shape[0]
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        cot_texts.append(text)
        cot_confidences.append(extract_final_confidence(text))

    if (i // BATCH_SIZE) % 20 == 0:
        temp_df = df.iloc[:len(cot_texts)].copy()
        temp_df['cot_reasoning'] = cot_texts
        temp_df['cot_confidence'] = cot_confidences
        temp_df.to_parquet(args.output_path.replace('.parquet', '_checkpoint.parquet'))

df['cot_reasoning'] = cot_texts
df['cot_confidence'] = cot_confidences

n_missing = sum(c is None for c in cot_confidences)
print(f"Could not parse a confidence value for {n_missing} / {len(df)} examples ({100*n_missing/len(df):.1f}%)")
df.to_parquet(args.output_path)
print(f"Saved to {args.output_path}")
