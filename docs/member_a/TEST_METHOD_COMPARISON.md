# Final Group Test Comparison

Status: **A/B/C/D final-model test results integrated**  
Dataset: **Dataset v1.0 FROZEN**  
Test split: **5,000 images, 500 classes, 10 images per class**

Only previously saved final-model test results are included. No model was
retrained and no test inference was rerun during this integration.

## Unified test metrics

| Member | Final method | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| A | Combined Colour + LBP + HOG + linear SVM objective | 4.48% | 12.16% | 4.90% | 4.48% | 3.99% |
| B | SIFT-BoVW + LinearSVM (K=2000) | 8.86% | 17.80% | 7.41% | 8.86% | 7.71% |
| C | C1 ResNet-18 Scratch + augmentation | 41.32% | 64.18% | 41.40% | 41.32% | 40.45% |
| D | ResNet-18 Pretrained | 65.10% | 83.34% | 66.01% | 65.10% | 64.42% |
| D | EfficientNet-B0 Pretrained | **76.66%** | **90.50%** | **77.63%** | **76.66%** | **76.20%** |

For this single-label classification task, overall accuracy equals Top-1.

## Validation-to-test gap

Gap is calculated as `test - validation` in percentage points.

| Member/method | Top-1 gap | Top-5 gap | Macro F1 gap |
|---|---:|---:|---:|
| A Combined + SVM | -0.32 pp | -0.38 pp | -0.26 pp |
| B SIFT-BoVW + SVM | -0.22 pp | -0.60 pp | -0.19 pp |
| C1 Scratch ResNet-18 + augmentation | +0.42 pp | +0.88 pp | +0.29 pp |
| D ResNet-18 Pretrained | +2.30 pp | +1.12 pp | +2.11 pp |
| D EfficientNet-B0 Pretrained | +0.70 pp | +0.56 pp | +0.60 pp |

The observed gaps are small enough that the relative ranking is stable across
validation and test. EfficientNet-B0 Pretrained is the strongest final model,
followed by ResNet-18 Pretrained and C1 Scratch ResNet-18.

Machine-readable values are stored in
`test_method_comparison.csv` in the same directory.
