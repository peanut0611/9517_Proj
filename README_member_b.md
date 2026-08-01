# Member B — SIFT + Bag-of-Visual-Words

Traditional local-feature pipeline for the COMP9517 group project.

## What this does

1. Extract **SIFT** descriptors from grayscale images  
2. Build a visual vocabulary with **MiniBatchKMeans** (train split only)  
3. Encode each image as a **BoVW** histogram (+ optional TF-IDF, L2 norm)  
4. Train **Linear SVM** and **Random Forest**  
5. Evaluate with the same metric fields as Member C (`top1`, `top5`, macro P/R/F1)

## Setup

```bash
cd /Users/peanut/Code/UNSW/COMP9517/9517_Proj
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-member-b.txt
```

Required local data (already frozen):

```text
Dataset_v1.0_FROZEN/train_mini/
Dataset_v1.0_FROZEN/val/
data_metadata/{train,val,test}.csv
```

## Run pilot (20 classes)

```bash
source .venv/bin/activate
python -m src.traditional.run_pipeline --config configs/b0_pilot_20cls.yaml
```

## Run full 500-class draft

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
  confusion_matrix_*.npy
  timing*.json
  model.joblib
random_forest/
  ...
```

## Ablations to run later

Change one factor at a time in the YAML configs:

- vocabulary size: 500 / 1000 / 2000  
- SVM vs Random Forest (already both trained each run)  
- TF-IDF on vs off  
- histogram L2 normalize on vs off  

Do not use the test set for choosing vocabulary size or classifier hyperparameters.
