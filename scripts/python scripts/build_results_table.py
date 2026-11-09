import json 
import pandas as pd 
import matplotlib.pyplot as plt 

MODELS = ["Pythia-410M", "Pythia-1.4B", "Pythia-2.8B", "Pythia-6.9B", "Qwen2.5-1.5B"] 

PATHS = { "Pythia-410M": "/workspace/project/results/pythia-410m_probe_results.json", "Pythia-1.4B": "/workspace/project/results/pythia-1.4b_probe_results.json", "Pythia-2.8B": "/workspace/project/results/pythia-2.8b_probe_results.json", "Pythia-6.9B": "/workspace/project/results/pythia-6.9b_probe_results.json", "Qwen2.5-1.5B": "/workspace/project/results/qwen2.5-1.5b_probe_results.json", }

all_results = []
for model in MODELS:
    with open(PATHS[model]) as f:
        data = json.load(f)
    for d in data:
        d['model'] = model  
    all_results.extend(data)
    

df = pd.DataFrame(all_results)
pivot = df.pivot_table(index=['model', 'feature'], columns='setting', values='auroc_mean').reset_index()
print(pivot.to_string(index=False))
pivot.to_csv("/workspace/project/results/final_auroc_table.csv", index=False)

features = df['feature'].unique()
colors = plt.cm.tab10.colors  

fig, ax = plt.subplots(figsize=(10, 6))
for i, feature in enumerate(features):
    sub = df[df['feature'] == feature]
    color = colors[i % len(colors)]

    ah = sub[sub['setting'] == 'AH-only'].set_index('model')['auroc_mean'].reindex(MODELS)
    uh = sub[sub['setting'] == 'UH-only'].set_index('model')['auroc_mean'].reindex(MODELS)

    ax.plot(MODELS, ah, marker='o', linestyle='-', color=color, label=f"{feature} (AH)")
    ax.plot(MODELS, uh, marker='s', linestyle='--', color=color, alpha=0.6, label=f"{feature} (UH)")

ax.axhline(0.5, color='gray', linestyle=':', label='chance')
ax.set_ylabel("AUROC")
ax.set_title("AH vs UH Detection AUROC Across Model Scale")
ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5)) 
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("/workspace/project/results/ah_auroc_across_scale.png", dpi=150, bbox_inches='tight')
print("Saved figure to /workspace/project/results/ah_auroc_across_scale.png")
