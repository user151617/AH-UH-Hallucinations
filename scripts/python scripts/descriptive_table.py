import pandas as pd

MODELS = {
    "Pythia-1.4B": "/workspace/project/results/pythia-1.4b_final_labeled.parquet",
    "Pythia-2.8B": "/workspace/project/results/pythia-2.8b_final_labeled.parquet",
    "Pythia-410M": "/workspace/project/results/pythia-410m_final_labeled.parquet",
    "Pythia-6.9B": "/workspace/project/results/pythia-6.9b_final_labeled.parquet",
    "Qwen2.5-1.5B": "/workspace/project/results/qwen2.5-1.5b_final_labeled.parquet",
}

rows = []
for model_name, path in MODELS.items():
    df = pd.read_parquet(path)
    counts = df['hallucination_label'].value_counts()
    total = len(df)
    rows.append({
        "Model": model_name,
        "N": total,
        "FA_n": counts.get("FA", 0),
        "FA_%": round(100 * counts.get("FA", 0) / total, 1),
        "AH_n": counts.get("AH", 0),
        "AH_%": round(100 * counts.get("AH", 0) / total, 1),
        "UH_n": counts.get("UH", 0),
        "UH_%": round(100 * counts.get("UH", 0) / total, 1),
    })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("/workspace/project/results/descriptive_table.csv", index=False)
print("\nSaved to /workspace/project/results/descriptive_table.csv")

print("\n--- Compare against anchor paper (fill in from Table 1 of arXiv:2510.09033) ---")
print("Llama-3-8B:  FA=__%  AH=__%  UH=__%")
print("Mistral-7B:  FA=__%  AH=__%  UH=__%")
