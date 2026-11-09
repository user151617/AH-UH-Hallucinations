import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "EleutherAI/pythia-6.9b"
INPUT_PATH = "/workspace/project/results/pythia-6.9b_labeled.parquet"
OUTPUT_PATH = "/workspace/project/results/pythia-6.9b_with_jsdiv.parquet"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   
    device_map="auto",
    attn_implementation="eager", 
)
model.eval()

def find_subject_token_positions(prompt, subject, tokenizer):
    """Map the subject string's character span to token indices."""
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

def get_output_distribution(model, input_ids, attention_mask, hooks=None):
    handles = []
    if hooks:
        for layer_module, hook_fn in hooks:
            handles.append(layer_module.register_forward_pre_hook(hook_fn, with_kwargs=True))
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
    finally:
        for h in handles:
            h.remove()
    return probs

def make_block_hook(subject_positions, last_position, seq_len, device, dtype):
    block_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    for pos in subject_positions:
        block_mask[0, 0, last_position, pos] = float("-inf")

    def hook(module, args, kwargs):
        am = kwargs.get("attention_mask", None)
        kwargs["attention_mask"] = block_mask if am is None else am + block_mask
        return args, kwargs
    return hook

def js_divergence(p, q, eps=1e-10):
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum()
    kl_qm = (q * (q / m).log()).sum()
    return (0.5 * kl_pm + 0.5 * kl_qm).item()

df = pd.read_parquet(INPUT_PATH)
jsdivs = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['question']
    subject = row['subj']

    subj_positions, encoding = find_subject_token_positions(prompt, subject, tokenizer)
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    seq_len = input_ids.shape[1]
    last_position = seq_len - 1

    if not subj_positions:
        jsdivs.append(None)  
        continue

    baseline_probs = get_output_distribution(model, input_ids, attention_mask, hooks=None)

    all_layers = model.gpt_neox.layers 
    hooks = [
        (layer.attention, make_block_hook(subj_positions, last_position, seq_len, model.device, torch.float32))
        for layer in all_layers
    ]
    blocked_probs = get_output_distribution(model, input_ids, attention_mask, hooks=hooks)

    jsdivs.append(js_divergence(baseline_probs, blocked_probs))

df['js_divergence'] = jsdivs
n_failed = df['js_divergence'].isna().sum()
print(f"Subject not found in {n_failed} / {len(df)} examples (excluded)")
df.to_parquet(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
