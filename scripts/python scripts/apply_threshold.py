import pandas as pd

INPUT_PATH = "/workspace/project/results/pythia-6.9b_with_jsdiv.parquet"
OUTPUT_PATH = "/workspace/project/results/pythia-6.9b_final_labeled.parquet"

df = pd.read_parquet(INPUT_PATH)
df = df.dropna(subset=['js_divergence'])  

fa_mask = df['is_correct'] == True
threshold = df.loc[fa_mask, 'js_divergence'].mean()
print(f"FA average JS-divergence (threshold): {threshold:.4f}")

def assign_label(row):
    if row['is_correct']:
        return 'FA'
    return 'AH' if row['js_divergence'] >= threshold else 'UH'

df['hallucination_label'] = df.apply(assign_label, axis=1)

print(df['hallucination_label'].value_counts())
print(df['hallucination_label'].value_counts(normalize=True))

df.to_parquet(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
