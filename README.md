# COMP9517 Group Project — iNaturalist Species Classification

Fine-grained species classification on a frozen 500-class subset of iNaturalist-2021.

**Code layout:** see [`docs/CODE_STRUCTURE.md`](docs/CODE_STRUCTURE.md).

## Code file structure

```text
9517_Proj/
├── notebooks/
│   ├── Member_A.ipynb      # handcrafted Colour+LBP+HOG
│   ├── Member_B.ipynb      # SIFT-BoVW + SVM/RF
│   ├── Member_C.ipynb      # ResNet-18 scratch
│   ├── Member_D.ipynb      # pretrained ResNet-18 / EfficientNet-B0
│   └── Member_E.ipynb      # Grad-CAM
├── src/
│   ├── data/               # A: splits / audit
│   ├── evaluation/         # A: shared NPZ evaluator
│   ├── traditional/        # B: SIFT-BoVW library
│   └── deep/               # C: CNN train/eval library
├── configs/                # B + C YAML configs
├── scripts/                # B CLI helpers
├── tests/
├── data_metadata/          # frozen CSVs (no images)
├── docs/                   # report/video notes by member
├── report/                 # CVPR LaTeX group report (main.tex → main.pdf)
├── requirements.txt
├── README.md
├── DATASET_README.md
└── EVALUATION_README.md
```

Local-only (not for ZIP): `Dataset_v1.0_FROZEN/`, `outputs/`, `experiment_workspace/`, `.venv/`.

## Quick start (Member B)

```bash
cd /path/to/9517_Proj
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

1. Place images under `Dataset_v1.0_FROZEN/` (see `DATASET_README.md`).
2. Open `notebooks/Member_B.ipynb` with Jupyter kernel **9517**.
3. Training switches default **off**; post-processing cells can re-export metrics / NPZ / cases.

Details: [`docs/member_b/README.md`](docs/member_b/README.md).

## CLI (Member B backend)

```bash
.venv/bin/python -m src.traditional.run_pipeline --config configs/b0_pilot_20cls.yaml
.venv/bin/python -m src.traditional.run_pipeline --config configs/b1_ablate_k2000.yaml
.venv/bin/python scripts/compare_vocab_ablation.py
```

## Final code ZIP (≤40 MB)

**Include:** `src/`, `configs/`, `scripts/`, `tests/`, `data_metadata/`, `notebooks/`, `docs/` (prefer text; drop large PNGs/PPT if over budget), root READMEs, `requirements.txt`.

**Exclude:** `Dataset_v1.0_FROZEN/`, `outputs/`, `experiment_workspace/`, `.venv/`, `*.pth`, `*.pt`, `*.joblib`, `*.npz`, videos.

## Group roles

| Role | Name | Focus |
|---|---|---|
| A | Xiangze Xue | data freeze; global handcrafted features |
| B | Zhigeng Qiao | SIFT-BoVW; repo structure / final ZIP checks |
| C | Zhiyang Ding | ResNet-18 from scratch |
| D | Yongqi Yang | pretrained CNNs; LaTeX report integration |
| E | Xinyan Chen | Grad-CAM and error analysis |

See also [`docs/group/COMP9517_group_workplan.md`](docs/group/COMP9517_group_workplan.md).
