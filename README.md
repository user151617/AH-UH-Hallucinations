# Internal-State Hallucination Detection Across Model Scale and Architecture

Code and results for an anonymous submission. This repository reproduces every number, table and figure in the paper.

The study tests whether the associated-hallucination (AH) versus unassociated-hallucination (UH) detection asymmetry reported in prior work holds across model scale and architecture. It also compares internal-state probes with verbalized and chain-of-thought (CoT) confidence.

## What is in here

| Folder / file | Contents |
|---|---|
| `scripts/` | All pipeline scripts (see "Pipeline" below) |
| `results/` | AUROC tables, ECE results, descriptive statistics, probe results (JSON) and figures |
| `requirements.txt`, `setup_env.sh` | Python dependencies and environment setup |
| `LICENSE` | MIT License |

Large intermediate files (generations and hidden-state features, in `.parquet` format) are not committed. They are recreated by running the pipeline.

## Models and data

| Model | Hugging Face ID | Architecture |
|---|---|---|
| Pythia-410M / 1.4B / 2.8B / 6.9B | `EleutherAI/pythia-*` | GPT-NeoX |
| Qwen2.5-1.5B (base checkpoint) | `Qwen/Qwen2.5-1.5B` | Qwen2 (grouped-query attention) |

- **Dataset:** PopQA (Mallen et al., 2023), stratified to 3,301 examples across 16 relation types.
- **Not redistributed:** model weights and the raw PopQA files. Download them from their official Hugging Face pages. Please follow their licenses.

## Setup

```bash
git clone <this repository>
cd <repository>
bash setup_env.sh          # or: pip install -r requirements.txt
```

Experiments were run on a single NVIDIA RTX 4090 (24 GB) with PyTorch and Hugging Face Transformers. All models run in float16.

## Pipeline

Run the scripts in this order, once per model. Each script's input and output paths and the model name are set by the arguments or constants at the top of the file.

1. `scripts/filter_popqa.py` loads PopQA and builds the stratified 3,301-example set.
2. `scripts/generate_responses.py` produces greedy factual completions.
3. `scripts/label_correctness.py` labels each response correct or incorrect (normalized substring match against the gold answer and its aliases).
4. `scripts/verify_subject_tokens.py` checks that subject-entity tokens are located correctly.
5. `scripts/causal_intervention.py` blocks attention from the subject tokens to the last token (all layers at once, in a single forward pass) and measures the Jensen-Shannon divergence of the output distribution.
6. `scripts/apply_threshold.py` assigns FA, AH or UH labels. The threshold is the mean divergence of the model's own FA examples.
7. `scripts/extract_features.py` extracts subject hidden states, last-token hidden states, subject-to-last-token attention, token probability and subject popularity.
8. `scripts/collect_verbalized_confidence.py` and `scripts/collect_cot_confidence.py` collect the two confidence-elicitation signals. The exact prompts are in Appendix A.2 of the paper and in these two files.
   ```bash
   python scripts/collect_verbalized_confidence.py --model_name <hf-id> --input_path <with_features.parquet> --output_path <out.parquet>
   python scripts/collect_cot_confidence.py        --model_name <hf-id> --input_path <with_features.parquet> --output_path <out.parquet>
   ```
9. `scripts/train_probes.py` trains the logistic-regression probes (standardized features, 70/30 split, 5 random seeds) and reports AH-only and UH-only AUROC.
10. `scripts/compute_ece.py`, `scripts/descriptive_table.py`, `scripts/norm_analysis.py` and `scripts/build_results_table.py` produce the ECE results, descriptive statistics, norm analysis, and the final table and figures.

## Main results

- Internal-state probes reach AUROC of about 0.79 to 0.92. AH detection is at least as good as UH detection in all five models.
- Verbalized confidence is close to chance (AUROC about 0.50).
- CoT confidence fails to produce a usable number for 90.7% to 94.5% of outputs on the four Pythia models. For Qwen2.5-1.5B it fails on 5.8%.
- Our causal intervention blocks all layers at once. The reference study intervenes layer by layer. This difference is a candidate explanation for the divergence from prior results, and it has not been tested.

Full tables are in `results/` and in the paper's appendix.

## Reproducibility notes

- Greedy decoding is used throughout, so generations are deterministic given the same model weights and library versions.
- Probes use 5 random seeds; results are reported as the mean over seeds.
- Run times and hardware are described in the paper.

## Citation

Citation details will be added after the review process.

## License

Released under the MIT License. See `LICENSE`.
