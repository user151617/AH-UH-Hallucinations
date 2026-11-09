import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_label", required=True, help="e.g. Pythia-1.4B (for output labeling)")
parser.add_argument("--features_path", required=True, help="*_with_features.parquet")
parser.add_argument("--confidence_path", required=True, help="*_with_confidence.parquet")
parser.add_argument("--output_path", required=True)
parser.add_argument("--n_seeds", type=int, default=5)
parser.add_argument("--cot_path", required=True, help="*_with_cot.parquet")
args = parser.parse_args()

df_feat = pd.read_parquet(args.features_path)
df_conf = pd.read_parquet(args.confidence_path)
df_cot = pd.read_parquet(args.cot_path)
df = df_feat.merge(df_conf[['id', 'verbalized_confidence']], on='id', how='left')
df = df.merge(df_cot[['id', 'cot_confidence']], on='id', how='left')
df = df[df['valid_features'] == True].copy()

FEATURE_BUILDERS = {
    "subject_hidden": lambda row: np.array(row['subj_hidden_features']),
    "last_token_hidden": lambda row: np.array(row['last_token_hidden_features']),
    "attention": lambda row: np.array(row['attn_subj_to_last_features']),
    "token_probability": lambda row: np.array([row['token_probability']]),
    "verbalized_confidence": lambda row: np.array([row['verbalized_confidence']]) if row['verbalized_confidence'] is not None else None,
    "cot_confidence": lambda row: np.array([row['cot_confidence']]) if row['cot_confidence'] is not None else None,
    "subject_popularity": lambda row: np.array([row['s_pop']]),
}

def build_dataset(df_subset, feature_name):
    X, y = [], []
    for _, row in df_subset.iterrows():
        feat = FEATURE_BUILDERS[feature_name](row)
        if feat is None or np.any(pd.isna(feat)):
            continue
        X.append(feat)
        y.append(1 if row['hallucination_label'] == 'FA' else 0)  
    return np.array(X), np.array(y)

results = []

for setting_name, halluc_label in [("AH-only", "AH"), ("UH-only", "UH")]:
    subset = df[df['hallucination_label'].isin(['FA', halluc_label])]

    for feature_name in FEATURE_BUILDERS:
        X, y = build_dataset(subset, feature_name)

        if len(X) < 20 or len(np.unique(y)) < 2:
            print(f"Skipping {setting_name} / {feature_name} - insufficient data ({len(X)} examples)")
            continue

        aurocs = []
        for seed in range(args.n_seeds):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, probs)
            aurocs.append(auroc)

        results.append({
            "model": args.model_label,
            "setting": setting_name,
            "feature": feature_name,
            "n_examples": len(X),
            "auroc_mean": np.mean(aurocs),
            "auroc_std": np.std(aurocs),
            "aurocs_all_seeds": aurocs,
        })
        print(f"{args.model_label} | {setting_name} | {feature_name}: "
              f"AUROC = {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f} (n={len(X)})")

with open(args.output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {args.output_path}")
