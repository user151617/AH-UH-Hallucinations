import pandas as pd
import re

INPUT_PATH = "/workspace/project/results/pythia-6.9b_generations.parquet"
OUTPUT_PATH = "/workspace/project/results/pythia-6.9b_labeled.parquet"

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  
    return text

def is_correct(generation, obj, possible_answers):
    gen_norm = normalize(generation)
    candidates = [obj] + list(possible_answers) if len(possible_answers) > 0 else [obj]
    for ans in candidates:
        if normalize(str(ans)) in gen_norm:
            return True
    return False

df = pd.read_parquet(INPUT_PATH)
df['is_correct'] = df.apply(
    lambda row: is_correct(row['generation'], row['obj'], row['possible_answers']),
    axis=1
)

print(f"Correct: {df['is_correct'].sum()} / {len(df)} ({df['is_correct'].mean():.1%})")
df.to_parquet(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
