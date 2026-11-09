import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "EleutherAI/pythia-6.9b"
INPUT_PATH = "/workspace/project/data/popqa_filtered.parquet"
OUTPUT_PATH = "/workspace/project/results/pythia-6.9b_generations.parquet"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 20

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"  
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} prompts")

questions = df['question'].tolist()
generations = []

for i in tqdm(range(0, len(questions), BATCH_SIZE)):
    batch = questions[i:i+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  
            pad_token_id=tokenizer.pad_token_id,
        )

    for j, output in enumerate(outputs):
        input_len = inputs['input_ids'][j].shape[0]
        gen_tokens = output[input_len:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        generations.append(gen_text)

    if (i // BATCH_SIZE) % 10 == 0:
        temp_df = df.iloc[:len(generations)].copy()
        temp_df['generation'] = generations
        temp_df.to_parquet(OUTPUT_PATH.replace('.parquet', '_checkpoint.parquet'))

df['generation'] = generations
df.to_parquet(OUTPUT_PATH)
print(f"Saved {len(df)} generations to {OUTPUT_PATH}")
