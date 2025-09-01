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
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose CUDA/CPU wheels as needed
pip install -U transformers peft accelerate datasets biopython pandas scikit-learn umap-learn matplotlib jupyter
# Optional if you use FAIR’s esm package directly in some notebooks:
pip install fair-esm
````

2. **Open notebooks**

```bash
jupyter notebook
```

3. **Run the pipeline**

* Start with `1-prep_hum_vir_metadata.ipynb` to load the included `SWP_*` files.
* Choose either:

  * **Zero/one-shot**: `4-ZeroShot-Infer.ipynb` / `4-One-ZeroShot-Infer-Copy.ipynb`
  * **Fine-tune** (LoRA/QLoRA): `3-TRAIN-lora-humvir.ipynb` or `new-esm-human-virus-finetune-*.ipynb`
* Evaluate with `MODEL-2-EvalDL-2024*.ipynb` and examine error characteristics with `analyze_humVir_mistakes+prep_iedb.ipynb`.

---

## Reproducing the main result

* Use the provided `SWP_hum_vir_*` train/test splits.
* For **ESM2** fine-tuning, follow the default hyperparameters in `3-TRAIN-lora-humvir.ipynb`.
* You should obtain ~**ROC-AUC ≈ 0.99** on held-out test proteins (with CV), and find that misclassified viral proteins are enriched for known/putative **immune-evasive** functions.

---

## Data
* Source curation followed Swiss-Prot **human** proteome and **vertebrate-host viral** proteins (see paper for details).
* For large-scale re-derivations, see paper Methods; you will need UniProt/Swiss-Prot if you want the static/precomputed T5 embeddings and IEDB website for MHC-I immunogenicity.

* Dataset of output predictions (`mistake_preds_4cv_650+len_lr_2024.csv.gz`) is in outputs and also in Huggingface Datasets (with data dictionary)
- 📂 Dataset: [Human–Virus Protein Mistake Predictions](https://huggingface.co/datasets/GrimSqueaker/ProteinHumVir), with 25,117 sequences and 20 descriptive features.

---

## Citing

Please cite the journal article (preferred), or optionally the ICML poster.

**APA**

> Ofer, D., & Linial, M. (2025). *Protein Language Models Expose Viral Immune Mimicry*. **Viruses, 17**(9), 1199. [https://doi.org/10.3390/v17091199](https://doi.org/10.3390/v17091199)

**BibTeX (journal)**

```bibtex
@Article{Ofer2025_Viruses,
  author        = {Ofer, Dan and Linial, Michal},
  title         = {Protein Language Models Expose Viral Immune Mimicry},
  journal       = {Viruses},
  year          = {2025},
  volume        = {17},
  number        = {9},
  article-number= {1199},
  doi           = {10.3390/v17091199},
  url           = {https://www.mdpi.com/1999-4915/17/9/1199}
}
```

**BibTeX (ICML 2024 ML4LMS poster)**

```bibtex
@inproceedings{OferLinial_ICML2024_ML4LMS,
  author    = {Ofer, Dan and Linial, Michal},
  title     = {Protein language models expose viral mimicry and immune escape},
  booktitle = {ICML 2024 Workshop on Machine Learning for Life and Material Science (ML4LMS)},
  year      = {2024},
  url       = {https://openreview.net/forum?id=gGnJBLssbb},
  note      = {Poster}
}
```

---

## Contact

* Issues/requests: GitHub Issues
* Authors: Dan Ofer, Michal Linial

```

pip install -U transformers peft accelerate datasets biopython pandas scikit-learn umap-learn matplotlib jupyter
# Optional if you use FAIR’s esm package directly in some notebooks:
pip install fair-esm
