import pandas as pd
import numpy as np
import json

def compute_ece(confidences, correctness, n_bins=10):
    """Expected Calibration Error - bins predictions by confidence, compares avg confidence to actual accuracy per bin."""
    confidences = np.array(confidences)
    correctness = np.array(correctness).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_details = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correctness[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_conf - bin_acc)
        bin_details.append({"range": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()), "avg_confidence": round(bin_conf, 3), "actual_accuracy": round(bin_acc, 3)})

    return ece, bin_details

MODELS = {
    "Pythia-410M": ("/workspace/project/results/pythia-410m_with_confidence.parquet", "/workspace/project/results/pythia-410m_final_labeled.parquet"),
    "Pythia-1.4B": ("/workspace/project/results/pythia-1.4b_with_confidence.parquet", "/workspace/project/results/pythia-1.4b_final_labeled.parquet"),
    "Pythia-2.8B": ("/workspace/project/results/pythia-2.8b_with_confidence.parquet", "/workspace/project/results/pythia-2.8b_final_labeled.parquet"),
    "Pythia-6.9B": ("/workspace/project/results/pythia-6.9b_with_confidence.parquet", "/workspace/project/results/pythia-6.9b_final_labeled.parquet"),
    "Qwen2.5-1.5B": ("/workspace/project/results/qwen2.5-1.5b_with_confidence.parquet", "/workspace/project/results/qwen2.5-1.5b_final_labeled.parquet"),
}

results = {}
for model, (conf_path, label_path) in MODELS.items():
    df_conf = pd.read_parquet(conf_path)
    df_label = pd.read_parquet(label_path)
    df = df_conf.merge(df_label[['id', 'is_correct']], on='id', how='left')
    df = df.dropna(subset=['verbalized_confidence', 'is_correct'])

    ece, bins = compute_ece(df['verbalized_confidence'].values, df['is_correct'].values)
    results[model] = {"ece": ece, "n": len(df), "bins": bins}
    print(f"{model}: ECE = {ece:.4f} (n={len(df)})")

with open("/workspace/project/results/ece_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to /workspace/project/results/ece_results.json")
