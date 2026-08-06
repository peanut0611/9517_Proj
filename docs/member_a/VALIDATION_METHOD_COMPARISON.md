# Provisional Group Validation Comparison

Status: **A/B/C/D main-model validation results integrated**  
Dataset: **Dataset v1.0 FROZEN**  
Split: **validation only, 5,000 images, 500 classes, 10 images per class**

## Unified metrics

| Member | Method | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| A | Combined Colour + LBP + HOG + linear SVM objective | 4.80% | 12.54% | 5.13% | 4.80% | 4.25% |
| B | SIFT-BoVW + LinearSVM (k=2000) | 9.08% | 18.40% | 7.50% | 9.08% | 7.90% |
| C | C0 ResNet-18 Scratch, no random augmentation | 21.90% | 43.38% | 24.31% | 21.90% | 22.11% |
| C | C1 ResNet-18 Scratch + augmentation | 40.90% | 63.30% | 41.58% | 40.90% | 40.17% |
| D | ResNet-18 Pretrained | 62.80% | 82.22% | 64.11% | 62.80% | 62.31% |
| D | EfficientNet-B0 Pretrained | **75.96%** | **89.94%** | **77.13%** | **75.96%** | **75.61%** |

For this single-label classification task, overall accuracy is identical to
Top-1 accuracy.

## Timing evidence

| Member/run | Recorded training time | Recorded validation inference time | Scope |
|---|---:|---:|---|
| A | 289.11 s | 0.0275 s | Classifier fit/predict only; cached handcrafted-feature extraction excluded. |
| B | 385.92 s | 0.0329 s | LinearSVC fit/decision function only; SIFT, vocabulary and BoVW encoding excluded. |
| C0 | 804.28 s | 2.0609 s | Training epochs only; inference is forward pass only and excludes image loading. |
| C1 | 2,091.68 s | 2.0935 s | Training epochs only; inference is forward pass only and excludes image loading. |
| D ResNet-18 | 2,689.36 s | 5.8865 s | Full training loop including epoch validation; inference loop includes data loading and forward pass. |
| D EfficientNet-B0 | 4,746.78 s | 6.0029 s | Full training loop including epoch validation; inference loop includes data loading and forward pass. |

**Timing caveat:** these values are not yet suitable for a direct end-to-end
speed comparison because the traditional-method records exclude feature
extraction and the deep-model records use a different timing boundary. B's
complete pipeline timing is still pending.

## Evidence-backed observations

- D's pretrained EfficientNet-B0 has the strongest validation result across
  every reported accuracy and macro metric.
- Adding the specified augmentation improved Scratch ResNet-18 Top-1 from
  21.90% to 40.90%, an absolute increase of 19.00 percentage points.
- B's SIFT-BoVW method improved Top-1 over A's global handcrafted descriptor
  from 4.80% to 9.08%, an absolute increase of 4.28 percentage points.
- Under the reported configurations, pretrained ResNet-18 improved Top-1 over
  augmented Scratch ResNet-18 from 40.90% to 62.80%, an increase of 21.90
  percentage points.
- EfficientNet-B0 improved Top-1 over pretrained ResNet-18 from 62.80% to
  75.96%, an increase of 13.16 percentage points.

## Timing limitation

Member B did not separately record the cold-start SIFT extraction time, so its
reported training and inference times should not be presented as complete
end-to-end pipeline timing. Test results are deliberately excluded from this
validation comparison.

Machine-readable values are stored in `validation_method_comparison.csv` in
the same directory.
