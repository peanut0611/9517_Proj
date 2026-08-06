# Code structure (submission-oriented)

This document lists **source code and notebooks** that belong in the final ≤40MB ZIP.  
Large local-only trees (`Dataset_v1.0_FROZEN/`, `outputs/`, `experiment_workspace/`) are omitted from the submission package.

```text
9517_Proj/
├── README.md
├── DATASET_README.md
├── EVALUATION_README.md
├── requirements.txt
│
├── notebooks/                          # member entry points
│   ├── Member_A.ipynb                  # A: handcrafted features + SGD/SVM
│   ├── Member_B.ipynb                  # B: SIFT-BoVW + SVM/RF
│   ├── Member_C.ipynb                  # C: ResNet-18 from scratch
│   ├── Member_D.ipynb                  # D: pretrained ResNet-18 / EfficientNet-B0
│   └── Member_E.ipynb                  # E: Grad-CAM analysis
│
├── src/
│   ├── __init__.py
│   ├── data/                           # A: freeze / audit / split tools
│   │   ├── audit_sha256.py
│   │   ├── audit_splits.py
│   │   ├── create_splits.py
│   │   ├── extract_subset.py
│   │   ├── prepare_category_repair.py
│   │   └── repair_category_dry_run.py
│   ├── evaluation/                     # A: shared NPZ evaluator
│   │   ├── __init__.py
│   │   └── evaluate_predictions.py
│   ├── traditional/                    # B: SIFT-BoVW library
│   │   ├── __init__.py
│   │   ├── classifiers.py
│   │   ├── common.py
│   │   ├── data.py
│   │   ├── encode.py
│   │   ├── metrics.py
│   │   ├── run_pipeline.py             # CLI: python -m src.traditional.run_pipeline
│   │   ├── sift.py
│   │   └── vocabulary.py
│   └── deep/                           # C: CNN train/eval utilities
│       ├── __init__.py
│       ├── analyze_errors.py
│       ├── common.py
│       ├── compare_runs.py
│       ├── data.py
│       ├── engine.py
│       ├── evaluate.py
│       ├── metrics.py
│       ├── model.py
│       └── train.py
│
├── configs/
│   ├── b0_pilot_20cls.yaml             # B pilot
│   ├── b1_full500_k1000.yaml           # B baseline K=1000
│   ├── b1_ablate_k500.yaml
│   ├── b1_ablate_k2000.yaml            # B selected final
│   ├── b1_rf_only_light.yaml
│   └── c1_scratch_aug_frozen_v1.yaml   # C1 frozen config
│
├── scripts/                            # B helpers (also covered in Member_B.ipynb)
│   ├── compare_vocab_ablation.py
│   ├── export_prediction_cases.py
│   └── export_shared_eval_scores.py
│
├── tests/
│   └── test_evaluate_predictions.py
│
├── data_metadata/                      # frozen relative-path CSVs (no images)
│   ├── train.csv / val.csv / test.csv
│   ├── class_to_idx.json
│   ├── taxonomy.csv
│   └── … audit / summary JSON …
│
└── docs/                               # report/video notes (text; figures optional)
    ├── group/
    ├── member_a/
    ├── member_b/
    ├── member_c/
    └── member_e/                       # Grad-CAM notes + PNG panels (omit PNGs if ZIP tight)
```

## Ownership map

| Member | Primary code | Entry |
|---|---|---|
| A | `src/data/`, `src/evaluation/`, `notebooks/Member_A.ipynb` | notebook |
| B | `src/traditional/`, `configs/b*.yaml`, `scripts/`, `notebooks/Member_B.ipynb` | notebook or `python -m src.traditional.run_pipeline` |
| C | `src/deep/`, `configs/c1_*.yaml`, `notebooks/Member_C.ipynb` | notebook |
| D | `notebooks/Member_D.ipynb` | notebook (PyTorch torchvision) |
| E | `notebooks/Member_E.ipynb` (+ `docs/member_e/` figures for report) | notebook |

## Local-only (do **not** put in code ZIP)

```text
Dataset_v1.0_FROZEN/          # images (~2.5GB)
outputs/                      # B SIFT cache + joblib models (~28GB)
experiment_workspace/         # D/E checkpoints & runtime (~200MB+)
.venv/
*.pth / *.pt / *.joblib / *.npz
```

## How to run (minimal)

```bash
# B traditional
.venv/bin/python -m src.traditional.run_pipeline --config configs/b1_ablate_k2000.yaml

# Shared evaluator unit test
python -m unittest discover -s tests -p 'test_evaluate_predictions.py' -v

# Or open notebooks/Member_{A,B,C,D,E}.ipynb with an appropriate kernel
```
