import pandas as pd
import numpy as np
import ast

RANDOM_SEED = 42
TARGET_PER_RELATION = 220
INPUT_PATH = "/workspace/project/data/popqa_raw.parquet"
OUTPUT_PATH = "/workspace/project/data/popqa_filtered.parquet"

np.random.seed(RANDOM_SEED)

df = pd.read_parquet(INPUT_PATH)
print(f"Raw dataset size: {len(df)}")
print(f"Relation types: {df['prop'].nunique()}")

required_cols = ['question', 'subj', 'obj', 'possible_answers', 'prop', 's_pop']
df = df.dropna(subset=required_cols)
print(f"After dropping missing fields: {len(df)}")

def parse_aliases(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return [x]
    return x

df['possible_answers'] = df['possible_answers'].apply(parse_aliases)
df = df[df['possible_answers'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
print(f"After filtering empty aliases: {len(df)}")

df = df.drop_duplicates(subset=['subj', 'prop'])
print(f"After deduplication: {len(df)}")

sampled_dfs = []
for relation, group in df.groupby('prop'):
    n = min(TARGET_PER_RELATION, len(group))
    sampled_dfs.append(group.sample(n=n, random_state=RANDOM_SEED))
    print(f"  {relation}: sampled {n} of {len(group)}")

filtered_df = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"\nFinal filtered dataset size: {len(filtered_df)}")

print("\nPopularity distribution (s_pop) after sampling:")
print(filtered_df['s_pop'].describe())

filtered_df.to_parquet(OUTPUT_PATH)
print(f"\nSaved to {OUTPUT_PATH}")


