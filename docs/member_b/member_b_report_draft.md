# Member B — Report Draft Notes (SIFT + BoVW)

Use these paragraphs when integrating into the group CVPR LaTeX report.
Numbers are from Dataset v1.0 (FROZEN), seed 9517.

## Suggested Methods text

We implement a classical local-feature pipeline based on SIFT descriptors and a
Bag-of-Visual-Words (BoVW) representation. Each image is converted to grayscale
and optionally resized so that its longer side is at most 512 pixels. Dense local
descriptors are extracted with OpenCV SIFT. A visual vocabulary is learned only
on the training split by MiniBatchKMeans clustering of sampled descriptors. Each
image is then encoded as a hard-assignment histogram over visual words, followed
by TF–IDF weighting and $\ell_2$ normalisation. The resulting vectors are
classified with a linear SVM (one-vs-rest LinearSVC). We also compare a capacity-
limited Random Forest on the same features.

This design complements Member A's global descriptors (colour / LBP / HOG): BoVW
emphasises repeatable local patterns rather than whole-image statistics, which is
relevant for fine-grained species cues but discards spatial layout.

## Suggested Experimental setup (Member B portion)

- Data: Dataset v1.0, 500 classes, 40/10/10 images per class.
- Vocabulary sizes compared: $K \in \{500, 1000, 2000\}$.
- Classifier: LinearSVC ($C=1$, `dual=false`); RF ablation used 100 trees,
  `max_depth=30` (unrestricted RF exceeded disk limits).
- Metrics: Top-1, Top-5, macro precision / recall / F1, train and inference time.
- Model selection: choose $K$ by validation macro-F1; report the corresponding
  held-out test metrics once.

## Results tables (copy into LaTeX)

### Main BoVW result (selected $K=2000$ on validation)

| Method | Top-1 | Top-5 | Macro-F1 | Train time | Test infer. (5k imgs) |
|---|---:|---:|---:|---:|---:|
| SIFT-BoVW + Linear SVM ($K=2000$) | 8.86% | 17.80% | 7.71% | ~386 s | ~0.04–0.2 s class |
| SIFT-BoVW + RF (limited) | 2.98%* | 7.90%* | 2.27%* | ~71 s* | ~0.16 s* |

\*RF numbers are from the $K=1000$ feature run with restricted tree depth; treat as a
capacity-limited reference, not a fully tuned forest.

Chance Top-1 for 500-way classification ≈ 0.20%.

### Vocabulary-size ablation (Linear SVM; all else fixed)

| $K$ | Val Top-1 | Val Macro-F1 | Test Top-1 | Test Top-5 | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| 500 | 8.28% | 6.54% | 8.82% | 18.68% | 7.07% |
| 1000 | 8.66% | 7.47% | 8.80% | 18.82% | 7.46% |
| **2000** | **9.08%** | **7.90%** | **8.86%** | 17.80% | **7.71%** |

Validation selects $K=2000$. Differences across $K$ are small, suggesting that
vocabulary size is not the main bottleneck compared with the limits of orderless
local histograms on fine-grained citizen-science photos.

## Suggested Discussion bullets

1. **Above chance, below CNNs.** ~9% Top-1 is far above 0.2% chance but far below
   Member C's scratch ResNet-18 (~41% Top-1). This matches expectations for
   classical BoVW on fine-grained data.
2. **Why BoVW struggles.** Orderless histograms discard pose and part layout;
   backgrounds and scale variation in iNaturalist images inject noisy SIFT words.
3. **SVM vs RF.** On the same histograms, linear SVM outperforms a depth-limited
   RF; a full deep forest was impractical due to model size on 500 classes.
4. **Ablation takeaway.** Increasing $K$ from 500 to 2000 yields only modest
   validation gains; report $K=2000$ as selected, without claiming large sensitivity.
5. **Cases.** See `docs/member_b/cases_k2000_svm/CASE_SHORTLIST.md` for high-confidence
   correct and incorrect examples to illustrate local-texture successes and
   fine-grained confusions.

## References to cite (starter list)

- Lowe, D. G. Distinctive image features from scale-invariant keypoints. *IJCV*, 2004. (SIFT)
- Csurka et al. Visual categorization with bags of keypoints. ECCV Workshop, 2004. (BoVW)
- Van Horn et al. The iNaturalist species classification and detection dataset. *CVPR*, 2018 / 2021 benchmark papers.
- Course lectures / OpenCV & scikit-learn documentation (implementation tools).

## Video talking points (~1.5–2 min)

1. Name + role: traditional local-feature method.
2. Pipeline slide: image → SIFT → vocabulary → histogram → SVM.
3. One ablation table ($K$) and the selected test numbers.
4. Show 1 success + 1 failure image from the case shortlist.
5. Takeaway: classical BoVW is a necessary baseline; deep models carry accuracy.
