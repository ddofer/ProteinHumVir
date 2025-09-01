# ProteinHumVir
Code & data for **Protein Language Models Expose Viral Immune Mimicry**.

> This repository contains the notebooks, data slices, and instructions used in our study showing that protein language models (PLMs) can robustly separate **viral** from **human** proteins—and that the *errors* they make are enriched for immune-evasive viral proteins with low predicted immunogenicity.

- 📄 Paper: https://www.mdpi.com/1999-4915/17/9/1199  
- 🧷 ICML 2024 (ML4LMS) poster: https://openreview.net/forum?id=gGnJBLssbb  
- 🔢 DOI: https://doi.org/10.3390/v17091199  


---

## TL;DR
- **Task:** binary classification of viral vs. human proteins using PLMs (ESM2/T5).  
- **Headline result:** ROC-AUC ≈ **0.997** on curated human/vertebrate-virus proteins.  
- **Key insight:** PLM “mistakes” concentrate on **immune mimicry** cases (e.g., viral IL-10, CD59-like proteins, TCR-β–like folds; cGAS–STING suppressors). In other words, the same mistakes as the biological immune system ("no adversarial examples", unlike in natural language or computer vision). 


---

## What’s in this repo?
Jupyter notebooks and compact data files to reproduce the main pipeline:

| Path | Purpose |
|---|---|
| `1-prep_hum_vir_metadata.ipynb` | Build/curate train/test tables from supplied Swiss-Prot–based slices. |
| `MODEL-1-basic_embed*.ipynb` | Compute embeddings (HF ESM2/T5) for sequences or FASTA batches. |
| `3-TRAIN-lora-humvir.ipynb`, `new-esm-human-virus-finetune-*.ipynb` | LoRA/QLoRA fine-tuning of ESM2 classifiers. |
| `4-ZeroShot-Infer.ipynb`, `4-One-ZeroShot-Infer-Copy.ipynb` | Zero-/one-shot inference baselines. |
| `MODEL-2-EvalDL-2024*.ipynb`, `cv-lora-humvir-2024.ipynb` | Evaluation, cross-validation, and metrics. |
| `analyze_humVir_mistakes+prep_iedb.ipynb` | Interpret errors; prep epitope/immunogenicity queries (IEDB-based). |
| `Cluster-Viz.ipynb`, `map_virus_lineage_baltimore.ipynb` | Visualization of embeddings/taxonomy patterns. |
| `SWP_*.csv.gz/tsv/fasta` | Pre-made human vs. vertebrate-virus sequence tables and FASTA used in the paper. |

> Note: The repository name reflects the preprint title; the **published** title is “Protein Language Models Expose Viral Immune Mimicry”.

---

## Quick start
* Not tested!
1) **Clone and set up**
```bash
git clone https://github.com/ddofer/ProteinHumVir
cd ProteinHumVir
# Create an environment (example with pip)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U torch --index-url https://download.pytorch.org/whl/cu121  # choose CUDA/CPU wheels as needed
pip install -U transformers peft accelerate datasets biopython pandas scikit-learn umap-learn matplotlib jupyter
# Optional if you use FAIR’s esm package directly in some notebooks:
pip install fair-esm
