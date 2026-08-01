# COMP9517 Group Project — iNaturalist Species Classification

Fine-grained species classification on a frozen 500-class subset of iNaturalist-2021.

## Repository contents

| Path | Description |
|---|---|
| `src/data/` | Dataset split / audit scripts (Member A) |
| `src/traditional/` | SIFT + Bag-of-Visual-Words + SVM/RF (Member B) |
| `configs/` | Experiment YAML configs |
| `scripts/` | Ablation comparison helpers |
| `data_metadata/` | Frozen split lists and class mapping (**no images**) |
| `docs/` | Results notes |
| `DATASET_README.md` | How to obtain and place the image archives |

Large files are **not** in git: images, model weights, SIFT caches, `outputs/`.

## Setup (Member B traditional pipeline)

```bash
cd /path/to/9517_Proj
uv venv .venv
source .venv/bin/activate   # or use .venv/bin/python directly
uv pip install -r requirements-member-b.txt
```

Place extracted images so paths in `data_metadata/*.csv` resolve:

```text
Dataset_v1.0_FROZEN/
  train_mini/
  val/
```

See `DATASET_README.md` for the official download and freeze rules.

## Run BoVW experiments

```bash
# Pilot (20 classes)
.venv/bin/python -m src.traditional.run_pipeline --config configs/b0_pilot_20cls.yaml

# Full 500-class baseline (K=1000)
.venv/bin/python -m src.traditional.run_pipeline --config configs/b1_full500_k1000.yaml

# Vocabulary ablations
.venv/bin/python -m src.traditional.run_pipeline --config configs/b1_ablate_k500.yaml
.venv/bin/python -m src.traditional.run_pipeline --config configs/b1_ablate_k2000.yaml

# Compare ablation metrics
.venv/bin/python scripts/compare_vocab_ablation.py
```

Resume classifier training from saved features:

```bash
.venv/bin/python -m src.traditional.run_pipeline \
  --config configs/b1_full500_k1000.yaml \
  --from-features
```

## Third-party libraries

- OpenCV (`opencv-python-headless`) — SIFT  
- scikit-learn — MiniBatchKMeans, LinearSVC, RandomForest  
- NumPy / Pandas / PyYAML / tqdm / joblib  

## Group roles (see `COMP9517_group_workplan.md`)

- A: data freeze; global handcrafted features  
- B: SIFT-BoVW; repo README / final ZIP checks  
- C: ResNet-18 from scratch  
- D: pretrained CNNs; LaTeX report integration  
- E: Grad-CAM and error analysis  
