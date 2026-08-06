# Member B Results Summary — SIFT + BoVW

**Date:** 2026-07-21  
**Dataset:** Dataset v1.0 - FROZEN (seed `9517`)  
**Method:** SIFT → MiniBatchKMeans vocabulary → BoVW (+ TF-IDF, L2) → Linear SVM / Random Forest

---

## 1. Official baseline (500 classes) — use this in the report

**Experiment:** `b1_full500_sift_bovw_k1000`  
**Output:** `outputs/b1_full500_sift_bovw_k1000/`

| Setting | Value |
|---|---|
| Classes | 500 |
| Images | train 20,000 / val 5,000 / test 5,000 |
| SIFT | grayscale, `max_side=512` |
| Vocabulary size \(K\) | 1000 (built on train only) |
| Descriptor sample for k-means | up to 500,000 |
| Encoding | TF-IDF + L2 normalize |
| SVM | `LinearSVC`, `C=1.0`, `dual=false` |
| RF (light) | 100 trees, `max_depth=30`, `max_features=sqrt` |

### Held-out test metrics

| Classifier | Top-1 | Top-5 | Overall Acc. | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **SVM (primary)** | **8.80%** | **18.82%** | **8.80%** | 7.07% | 8.80% | **7.46%** |
| Random Forest (light) | 2.98% | 7.90% | 2.98% | 2.14% | 2.98% | 2.27% |

### Validation metrics (for model comparison only)

| Classifier | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| SVM | 8.66% | 19.18% | 7.47% |
| Random Forest (light) | 2.78% | 7.36% | 1.99% |

### Timing

| Stage / model | Time |
|---|---|
| SVM train | ~538 s (~9.0 min) |
| RF train (light) | ~71 s |
| SVM test inference (5,000 images) | ~0.04 s |
| RF test inference (5,000 images) | ~0.16 s |

Chance baseline for 500-way Top-1 ≈ **0.20%**. SVM is ~44× chance.

### Notes for the report

- Prefer **SVM** as the main traditional local-feature result.
- RF was capped (`n_estimators=100`, `max_depth=30`) after an unrestricted RF attempt ballooned to ~18GB and filled the disk. State this clearly so the RF number is not over-interpreted.
- Member C Scratch ResNet-18 + aug is ~41% Top-1 for context; BoVW being much lower is expected for this fine-grained setting.

---

## 2. Pilot (sanity check only — not for final tables)

**Experiment:** `b0_pilot_20cls_sift_bovw`  
**Output:** `outputs/b0_pilot_20cls_sift_bovw/`

| Setting | Value |
|---|---|
| Classes | 20 (seed-selected subset) |
| Images | 800 / 200 / 200 |
| Vocabulary \(K\) | 200 |

### Test metrics

| Classifier | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| SVM | 45.0% | 80.5% | 41.6% |
| Random Forest | 38.5% | 77.5% | 35.8% |

Pilot only proves the pipeline works. Do **not** compare these numbers directly to the 500-class baseline.

---

## 3. Artifact checklist

```text
outputs/b1_full500_sift_bovw_k1000/
├── summary.json                 # SVM + RF merged metrics
├── experiment_config.json
├── vocabulary.joblib
├── features.npz                 # BoVW features (train/val/test)
├── idf.npy
├── svm/
│   ├── metrics_{val,test}.json
│   ├── predictions_{val,test}.csv
│   ├── confusion_matrix_{val,test}.npy
│   ├── timing*.json
│   └── model.joblib
└── random_forest/
    └── (same layout)
```

SIFT descriptor cache (reusable for ablations): `outputs/cache/sift_full500/`.

---

## 4. Vocabulary-size ablation (SVM, all else fixed)

Outputs: `outputs/b1_ablate_k500/`, `outputs/b1_full500_sift_bovw_k1000/`, `outputs/b1_ablate_k2000/`  
Table CSV: `outputs/ablation_vocabulary_svm.csv`

| \(K\) | Val Top-1 | Val Top-5 | Val Macro-F1 | Test Top-1 | Test Top-5 | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| 500 | 8.28% | 18.72% | 6.54% | 8.82% | 18.68% | 7.07% |
| 1000 | 8.66% | 19.18% | 7.47% | 8.80% | 18.82% | 7.46% |
| **2000** | **9.08%** | 18.40% | **7.90%** | **8.86%** | 17.80% | **7.71%** |

Selection rule: best **validation macro-F1** → **\(K=2000\)**.  
Held-out test for the selected setting: Top-1 **8.86%**, Top-5 **17.80%**, macro-F1 **7.71%**.

Notes:
- Gains across \(K\) are small; BoVW remains far below CNNs.
- \(K=1000\) is still a reasonable baseline (best val Top-5).
- For the report, state that \(K\) was chosen on val, then quote the corresponding test numbers.

---

## 5. Still TODO for Member B

- [x] Vocabulary-size ablation on val (`K = 500 / 1000 / 2000`)
- [x] Success / failure case shortlist (`docs/member_b/cases_k2000_svm/`)
- [x] Report draft notes (`docs/member_b/member_b_report_draft.md`)
- [ ] Optional: TF-IDF on/off ablation
- [ ] Polish report text with D into final LaTeX
- [ ] Video segment: pipeline + ablation conclusions (use draft talking points)
- [ ] Final ZIP packaging check near deadline


---

## 6. One-line takeaway

On Dataset v1.0 (500 species), SIFT-BoVW + Linear SVM is ~**8.8–8.9%** Top-1 (~**7.5–7.7%** macro-F1). Validation selects **\(K=2000\)** (test 8.86% / 17.80% / 7.71%); vocabulary size has only a small effect. Far above chance, well below CNNs; capacity-limited RF underperforms SVM.
