# Member B → Shared Evaluation Handoff

Generated for Member A's common evaluator (`EVALUATION_README.md`).

## What to upload to OneDrive

Upload this folder (or the two NPZ files) to the group OneDrive:

```text
outputs/shared_eval_exports/
├── member_b_sift_bovw_svm_k2000_val_scores.npz     # validation (required for comparison)
├── member_b_sift_bovw_svm_k2000_test_scores.npz    # frozen one-shot test
└── member_b_sift_bovw_svm_k2000_export_manifest.json
```

Suggested OneDrive name:

```text
Member_B_sift_bovw_svm_k2000_shared_eval_2026-08-01/
```

## NPZ contents (per A’s format)

| Key | Shape / type | Notes |
|---|---|---|
| `scores` | `(5000, 500)` float | LinearSVC decision scores (higher = better rank) |
| `true_labels` | `(5000,)` int | `class_index` from frozen CSV |
| `image_paths` | `(5000,)` str | relative paths matching `data_metadata/{val,test}.csv` order |
| `class_indices` | `(500,)` int | `0..499` column order |

## Timing definitions (state when sharing)

- **training_seconds = 385.92**: `LinearSVC.fit` only on BoVW features  
  (excludes SIFT extraction / vocabulary / encoding)
- **validation inference_seconds = 0.0329**: scoring 5000 val images  
- **test inference_seconds = 0.0375**: scoring 5000 test images  

## Verified with shared evaluator

| Split | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| validation | 9.08% | 18.40% | 7.90% |
| test (frozen once) | 8.86% | 17.80% | 7.71% |

Local evaluator outputs (optional to share):

```text
outputs/shared_eval_exports/member_b_sift_bovw_svm_k2000_validation/
outputs/shared_eval_exports/member_b_sift_bovw_svm_k2000_test/
```

## How this was produced

```bash
export PYTHONPATH=.
.venv/bin/python scripts/export_shared_eval_scores.py
```

Model: selected vocabulary **K=2000** SVM under `outputs/b1_ablate_k2000/`.
