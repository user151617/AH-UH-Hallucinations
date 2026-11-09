import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True, help="e.g. EleutherAI/pythia-1.4b")
parser.add_argument("--input_path", required=True, help="path to *_final_labeled.parquet")
parser.add_argument("--output_path", required=True)
parser.add_argument("--layers_attr", default="auto", help="'gpt_neox' or 'model' - auto-detects if not set")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    attn_implementation="eager",
    output_hidden_states=True,
    output_attentions=True,
)
model.eval()

if hasattr(model, "gpt_neox"):
    num_layers = len(model.gpt_neox.layers)
elif hasattr(model, "model"):
    num_layers = len(model.model.layers)
else:
    raise ValueError("Unrecognized architecture - check model structure manually")

mid_start = num_layers // 3
mid_end = 2 * num_layers // 3
print(f"Model has {num_layers} layers. Using middle layers {mid_start}-{mid_end} for subject hidden states.")

def find_subject_token_positions(prompt, subject, tokenizer):
    encoding = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding["offset_mapping"][0].tolist()
    char_start = prompt.find(subject)
    if char_start == -1:
        return [], encoding
    char_end = char_start + len(subject)
    positions = [
        idx for idx, (s, e) in enumerate(offsets)
        if s != e and s < char_end and e > char_start
    ]
    return positions, encoding

df = pd.read_parquet(args.input_path)

subj_hidden_features = []
last_token_hidden_features = []
attn_subj_to_last_features = []
token_prob_features = []
valid_mask = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['question']
    subject = row['subj']

    subj_positions, encoding = find_subject_token_positions(prompt, subject, tokenizer)
    if not subj_positions:
        valid_mask.append(False)
        subj_hidden_features.append(None)
        last_token_hidden_features.append(None)
        attn_subj_to_last_features.append(None)
        token_prob_features.append(None)
        continue

    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    last_position = input_ids.shape[1] - 1

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    subj_vecs = []
    for layer_idx in range(mid_start, mid_end):
        layer_hidden = outputs.hidden_states[layer_idx][0]  
        subj_vec = layer_hidden[subj_positions].mean(dim=0)  
        subj_vecs.append(subj_vec)
    subj_hidden_concat = torch.cat(subj_vecs).cpu().numpy()  

    last_hidden = outputs.hidden_states[-1][0, last_position, :].cpu().numpy()

    attn_per_layer = []
    for layer_attn in outputs.attentions:  
        attn_slice = layer_attn[0, :, last_position, subj_positions]  
        attn_per_layer.append(attn_slice.mean().item())  
    attn_vec = np.array(attn_per_layer)

    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_prob = probs.max().item()

    valid_mask.append(True)
    subj_hidden_features.append(subj_hidden_concat)
    last_token_hidden_features.append(last_hidden)
    attn_subj_to_last_features.append(attn_vec)
    token_prob_features.append(top_prob)

df['valid_features'] = valid_mask
df['subj_hidden_features'] = subj_hidden_features
df['last_token_hidden_features'] = last_token_hidden_features
df['attn_subj_to_last_features'] = attn_subj_to_last_features
df['token_probability'] = token_prob_features

n_dropped = (~df['valid_features']).sum()
print(f"Dropped {n_dropped} / {len(df)} examples (subject tokens not found)")

df.to_parquet(args.output_path)
print(f"Saved to {args.output_path}")
