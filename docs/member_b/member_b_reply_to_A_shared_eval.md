# Reply to Member A — SIFT-BoVW final model disclosure

**From:** Member B  
**Model:** SIFT + Bag-of-Visual-Words + Linear SVM  
**Selected setting:** vocabulary size **K = 2000**  
**Dataset:** Dataset v1.0 – FROZEN, seed **9517**  
**Date:** 2026-08-03  

---

## 1. Final model configuration

### Image preprocessing
- Read as **grayscale** (`cv2.IMREAD_GRAYSCALE`)
- If longer side > **512**, resize with `cv2.INTER_AREA` so `max(h, w) = 512`
- No colour jitter, no random crop, no flip (handcrafted pipeline)

### SIFT parameters
- OpenCV `cv2.SIFT_create`
- `nfeatures = 0` (OpenCV default = unlimited)
- `contrastThreshold = 0.04`
- Other SIFT kwargs left at OpenCV defaults
- Descriptors: 128-D float32; images with no keypoints → empty descriptor set (histogram all zeros)

### Descriptor sampling (for vocabulary only)
- **Train split only**
- Shuffle training images with NumPy RNG seed **9517**
- Collect up to **500,000** descriptors (per-image take-all until budget filled; if an image has more than remaining budget, random subset without replacement)
- Actual sampled count for K=2000 run: **500,000**

### KMeans / vocabulary
- `sklearn.cluster.MiniBatchKMeans`
- `n_clusters = K = 2000` (selected)
- `random_state = 9517`
- `batch_size = 8192`
- `max_iter = 100`
- `n_init = "auto"`
- `reassignment_ratio = 0.01`
- Assignment: **hard** assignment (`predict`), then count histogram

### BoVW normalization
- Raw hard-assignment counts
- **TF–IDF** fitted on **training** histograms only, then applied to val/test
- **L2 normalize** each image histogram after TF–IDF

### Classifier (final)
- `sklearn.svm.LinearSVC`
- `C = 1.0`
- `max_iter = 8000`
- `dual = False`
- `class_weight = None` (default; **not** balanced)
- `random_state = 9517`

### Random seed summary
- Experiment / sampling / KMeans / LinearSVC: **9517**
- Same frozen class list and splits as the group Dataset v1.0

---

## 2. Timing record (K=2000 final SVM run)

Times below are wall-clock seconds from `outputs/b1_ablate_k2000/`.

| Stage | Seconds | Notes |
|---|---:|---|
| SIFT extraction (train) | 17.54 | **Cache hit** in this run (descriptors already on disk). First cold extraction for full 500-class set was earlier and **not separately archived** for this config → treat cold extract as **未单独归档**; do not use 17.54s as cold-extract cost. |
| SIFT extraction (val) | 4.20 | Same: cache hit |
| SIFT extraction (test) | 4.34 | Same: cache hit |
| Vocabulary build (sample + MiniBatchKMeans) | 16.71 | Recorded |
| BoVW encode (train+val+test together) | 163.76 | Recorded as one combined `encode_sec` (not split by subset) |
| Classifier training (LinearSVC.fit) | 385.92 | Recorded |
| Validation inference (decision scores, 5000 imgs) | 0.0329 | Recorded |
| Test inference (decision scores, 5000 imgs) | 0.0375 | Recorded |

**未记录 / 需说明的部分**
- Cold-start SIFT wall time for the full 30k images under K=2000: **未单独归档**
- Encode time broken down into train / val / test separately: **未记录**（只有合计 `encode_sec`）
- End-to-end “image path → prediction” latency including SIFT+encode at test time: **未记录**（提交给统一评估的 inference 仅为已编码特征上的 SVM scoring）

Timing definition used in shared-eval NPZ manifest:
- `training_seconds` = LinearSVC.fit only  
- `inference_seconds` = scoring already-encoded BoVW features only  

---

## 3. Written confirmation (protocol)

I confirm that:

1. **K = 2000 and all listed hyperparameters were chosen using the validation split only** (vocabulary-size ablation compared val macro-F1 / related val metrics; final K selected by **validation macro-F1**).
2. The model configuration was **frozen before the held-out test evaluation**.
3. The **test set was evaluated once** for the selected K=2000 Linear SVM (the scores NPZ already shared).
4. **After seeing test results, I did not further tune** K, SIFT settings, encoding, or SVM hyperparameters for the final model.

Ablation experiments also produced test numbers historically, but **model selection did not use test**; per your request, ablation comparison for the report should rely on **validation** metrics/scores only.

---

## 4. Extra experiments for the report (validation only)

### 4.1 Vocabulary-size ablation (Linear SVM; all else fixed)

| K | Val Top-1 | Val Top-5 | Val Macro-P | Val Macro-R | Val Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 500 | 8.28% | 18.72% | 6.01% | 8.28% | 6.54% |
| 1000 | 8.66% | 19.18% | 7.14% | 8.66% | 7.47% |
| **2000** | **9.08%** | 18.40% | **7.50%** | **9.08%** | **7.90%** |

Selection rule: best **validation macro-F1** → **K=2000**.

Optional raw validation score NPZs (same schema as before):

```text
outputs/shared_eval_exports/member_b_sift_bovw_svm_k500_val_scores.npz
outputs/shared_eval_exports/member_b_sift_bovw_svm_k1000_val_scores.npz
outputs/shared_eval_exports/member_b_sift_bovw_svm_k2000_val_scores.npz   # already shared
```

### 4.2 Classifier comparison on K=1000 BoVW features (validation)

Same BoVW features (K=1000, TF–IDF + L2). RF is a **capacity-limited** reference (disk constraints), not a fully tuned forest.

| Classifier | Val Top-1 | Val Top-5 | Val Macro-F1 | Train sec | Val infer sec |
|---|---:|---:|---:|---:|---:|
| Linear SVM (C=1, max_iter=8000, dual=False, class_weight=None) | 8.66% | 19.18% | 7.47% | 538.04 | (see svm timing_val if needed) |
| Random Forest (n_estimators=100, max_depth=30, max_features=sqrt, class_weight=None, random_state=9517) | 2.78% | 7.36% | 1.99% | 70.71 | ~0.16 |

Optional RF validation scores:

```text
outputs/shared_eval_exports/member_b_sift_bovw_rf_light_k1000_val_scores.npz
```

Please treat RF as an ablation/reference with restricted capacity, not as a competing fully optimized classical model.

---

## 5. Files already / newly available for you

Final selected model (already checked by you):

- `member_b_sift_bovw_svm_k2000_val_scores.npz`
- `member_b_sift_bovw_svm_k2000_test_scores.npz`

Additional for unified tables (validation only):

- `member_b_sift_bovw_svm_k500_val_scores.npz`
- `member_b_sift_bovw_svm_k1000_val_scores.npz`
- `member_b_sift_bovw_rf_light_k1000_val_scores.npz`

I can upload the extra validation NPZs to OneDrive in:

`Member_B_sift_bovw_ablation_val_scores_2026-08-03/`

---

Please tell me if you need any field renamed to match your evaluator’s expected metadata keys.
