# Scratch ResNet-18 and Data-Augmentation Ablation

Scratch-trained ResNet-18 on the frozen 500-class iNaturalist subset, with a
controlled augmentation ablation (**C0** no random aug vs **C1** crop/flip).

Notebook: [`../../notebooks/Member_C.ipynb`](../../notebooks/Member_C.ipynb)

## Report merge (from OneDrive, 2026-08-06)

| File | Purpose |
|---|---|
| [`member_c_report_merge.md`](member_c_report_merge.md) | Literature / Methods / Results / Discussion prose for the group CVPR report |
| [`member_c_references.bib`](member_c_references.bib) | BibTeX (He, Shorten, Loshchilov) |
| [`figures/`](figures/) | C0 vs C1 learning curves (PDF/PNG) |

See also [`SYNC_FROM_ONEDRIVE.md`](SYNC_FROM_ONEDRIVE.md).

## Setup

```bash
python -m pip install -r requirements.txt
```

Needs `data_metadata/` manifests and (for full runs) images under the frozen
dataset tree. Shared eval: `src/evaluation/evaluate_predictions.py`. Checkpoints
are not in the code ZIP.

## Held-out test results

| Experiment | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| C0: no random augmentation | 21.62% | 44.24% | 21.42% |
| C1: standard augmentation | 41.32% | 64.16% | 40.45% |

C1 improves Top-1 by **19.70** percentage points (seed 9517 only). Shared
evaluator Top-5 for C1 is **64.18%** (rank-5 tie); frozen JSON keeps **64.16%**.
