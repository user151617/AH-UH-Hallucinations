import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MODEL_LABEL = "Pythia-410M"
INPUT_PATH = "/workspace/project/results/pythia-410m_with_features.parquet"
OUTPUT_FIG = "/workspace/project/results/pythia-410m_subject_norms.png"

df = pd.read_parquet(INPUT_PATH)
df = df[df['valid_features'] == True].copy()

df['subj_hidden_norm'] = df['subj_hidden_features'].apply(lambda x: np.linalg.norm(np.array(x)))

summary = df.groupby('hallucination_label')['subj_hidden_norm'].agg(['mean', 'std', 'count'])
print(summary)

fig, ax = plt.subplots(figsize=(6, 5))
labels = ['FA', 'AH', 'UH']
means = [df[df['hallucination_label'] == l]['subj_hidden_norm'].mean() for l in labels]
stds = [df[df['hallucination_label'] == l]['subj_hidden_norm'].std() for l in labels]
ax.bar(labels, means, yerr=stds, capsize=5, color=['#4C72B0', '#DD8452', '#55A868'])
ax.set_ylabel("Subject hidden state norm (middle layers)")
ax.set_title(f"{MODEL_LABEL}: Subject representation norm by label")
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=150)
print(f"Saved to {OUTPUT_FIG}")
