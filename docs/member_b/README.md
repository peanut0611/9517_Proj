# Member B — SIFT + Bag-of-Visual-Words

Traditional local-feature pipeline for the COMP9517 group project.

**Primary entry:** [`../../notebooks/Member_B.ipynb`](../../notebooks/Member_B.ipynb) (kernel **9517**) — training, vocab ablation table, shared-eval NPZ export, and case shortlist in one place. Core library remains `src/traditional/` so Member A handoffs stay valid.

## What this does

1. Extract **SIFT** descriptors from grayscale images  
2. Build a visual vocabulary with **MiniBatchKMeans** (train split only)  
3. Encode each image as a **BoVW** histogram (+ optional TF-IDF, L2 norm)  
4. Train **Linear SVM** and **Random Forest**  
5. Evaluate with the same metric fields as Member C (`top1`, `top5`, macro P/R/F1)

## Setup

```bash
cd /path/to/9517_Proj
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or use the conda env registered as Jupyter kernel `9517`.

Required local data (already frozen):

```text
Dataset_v1.0_FROZEN/train_mini/
Dataset_v1.0_FROZEN/val/
data_metadata/{train,val,test}.csv
```

## Run via notebook (recommended)

1. Open `notebooks/Member_B.ipynb` with kernel **9517**
2. In the setup cell, leave `RUN_*` training flags `False` unless you intend to retrain
3. Run all cells for ablation table / NPZ export / cases (cheap if `outputs/` already exist)
4. To retrain: set e.g. `RUN_ABLATE_K2000 = True` (and `FROM_FEATURES` as needed)

## CLI alternative (same backend)

```bash
source .venv/bin/activate
python -m src.traditional.run_pipeline --config configs/b0_pilot_20cls.yaml
```

```bash
# Prevent macOS idle sleep while training
caffeinate -dimsu python -m src.traditional.run_pipeline --config configs/b1_full500_k1000.yaml
```

Full extraction is slow the first time; SIFT descriptors are cached under `outputs/cache/`.

If the run stops after features are saved (e.g. machine slept), resume classifier training only:

```bash
caffeinate -dimsu python -m src.traditional.run_pipeline \
  --config configs/b1_full500_k1000.yaml \
  --from-features
```

## Docs in this folder

| File | Purpose |
|---|---|
| `member_b_results_summary.md` | Metrics / ablation table |
| `member_b_report_draft.md` | Report prose draft for D |
| `member_b_video_script_2min.md` | ~2 min video script |
| `MemberB_Video_Slides.pptx` | Video slides |
| `member_b_reply_to_A_shared_eval.md` | Config + timings for A |
| `cases_k2000_svm/` | Success / failure shortlist |

## Outputs

Each run writes to `outputs/<experiment_name>/`:

```text
experiment_config.json
timing.json
summary.json
environment.txt
vocabulary.joblib
features.npz
svm/
  metrics_val.json
  metrics_test.json
  predictions_*.csv
  ...
```

## Ablations

Change one factor at a time in the YAML configs under `configs/`:

- vocabulary size: 500 / 1000 / 2000  
- SVM vs Random Forest  
- TF-IDF on vs off  
- histogram L2 normalize on vs off  

Do not use the test set for choosing vocabulary size or classifier hyperparameters.
