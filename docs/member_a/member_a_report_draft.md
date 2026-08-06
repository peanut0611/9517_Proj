# Member A Report Draft

Prepared for integration into the group CVPR report. This draft covers the
dataset, common experimental protocol, handcrafted global-feature baseline,
Member A ablations, final test result, and limitations. It should be shortened
and merged with the other members' sections rather than inserted as a separate
standalone report.

## 1. Dataset and split

We constructed a reproducible 500-species subset of iNaturalist 2021
(iNat2021) using random seed 9517. For every selected species, 40 images from
the official `train_mini` split were used for training and the remaining 10
were used for validation. The 10 corresponding images from the official
iNat2021 validation split were reserved as the held-out test set. This produced
20,000 training images, 5,000 validation images, and 5,000 test images, with
exactly balanced class frequencies. The official validation split is referred
to as the test set throughout this report, following the project
specification.

Each species was assigned a contiguous `class_index` from 0 to 499, while the
original iNaturalist category identifier and taxonomic fields (kingdom,
phylum, class, order, family, genus, and species) were retained. The split was
audited for image readability, missing paths, class counts, mapping
consistency, image-ID overlap, and byte-level duplicate content using SHA-256.
The audit identified one byte-identical image across the initial training and
test splits. The affected category was deterministically replaced while
preserving the class-index mapping, after which both the structural and
content-level audits passed. All methods in the group use this frozen Dataset
v1.0 and the same relative-path CSV manifests.

## 2. Common evaluation protocol

Hyperparameters were selected using the validation set only. The test set
remained inaccessible until the complete image preprocessing, descriptor, and
classifier configuration had been frozen. The final test set was then
evaluated once, without subsequent tuning.

All methods are compared using Top-1 accuracy, Top-5 accuracy, macro-averaged
precision, recall, and F1. Macro metrics give equal weight to every species and
are therefore important for this 500-class fine-grained task. We additionally
record training time, feature-extraction time where applicable, and inference
time. For this single-label task, overall accuracy is identical to Top-1
accuracy. A shared score-based evaluation interface was implemented to ensure
identical metric definitions and output formats across handcrafted and deep
models. It consumes an \(N \times 500\) score matrix and produces standardized
metrics, per-image Top-5 predictions, and a 500-class confusion matrix.

## 3. Handcrafted global-feature baseline

The traditional baseline represents each image using global colour, texture,
and shape information. All images are resized to \(128 \times 128\) pixels in
the final configuration.

**Colour descriptor.** We compute a 16-bin histogram for each HSV channel and
append the mean, standard deviation, and skewness of each RGB channel. This
gives a 57-dimensional vector that captures global colour distribution and
low-order colour statistics.

**Local Binary Pattern (LBP).** Uniform LBP with \(P=8\) sampling points and
radius \(R=1\) is computed on the grayscale image. A 10-bin normalized
histogram is formed in each cell of a \(4 \times 4\) spatial grid, producing a
160-dimensional texture descriptor. LBP provides local texture information
while retaining coarse spatial layout \cite{ojala2002multiresolution}.

**Histogram of Oriented Gradients (HOG).** HOG is extracted from the grayscale
image using 9 orientation bins, \(16 \times 16\)-pixel cells, and \(2 \times 2\)
cells per block with L2-Hys block normalization. At \(128 \times 128\), the HOG
vector has 1,764 dimensions. HOG captures local edge orientation and shape
while providing some robustness to illumination variation
\cite{dalal2005histograms}.

**Combined representation.** The final 1,981-dimensional representation is
the concatenation of the 57-dimensional colour descriptor, 160-dimensional
LBP descriptor, and 1,764-dimensional HOG descriptor.

The feature vector is standardized using statistics learned from the training
set. Classification uses a multiclass linear hinge-loss classifier, i.e. a
linear SVM objective optimized using stochastic gradient descent. The
regularization parameter is mapped using
\(\alpha=1/(C N_{\mathrm{train}})\). The final setting uses \(C=1\),
\(\alpha=5\times10^{-5}\), averaged SGD, tolerance \(10^{-3}\), a maximum of
1,000 epochs, eight CPU workers, and random seed 9517. Optimization stopped
after 175 epochs. This scalable formulation was used because direct
one-vs-rest linear SVM fitting was impractically slow for the 20,000-image,
500-class problem. Support vector classification and the maximum-margin
principle originate from \cite{cortes1995support}; the implementation uses
scikit-learn \cite{pedregosa2011scikit}.

## 4. Validation ablations

We varied one design decision at a time. The 20-class pilot was used only to
verify implementation correctness and is excluded from the final comparative
claims.

### 4.1 Input-resolution ablation

With HOG and \(C=1\) fixed, \(128 \times 128\) outperformed \(224 \times 224\)
in Top-1 accuracy and macro F1, while requiring approximately one third of the
training time. The paired Top-1 difference was \(+0.52\) percentage points
(95% CI: \(+0.06\) to \(+0.98\)); McNemar's test gave \(p=0.033\). We therefore
selected \(128 \times 128\).

| Resolution | Feature dimension | Top-1 | Top-5 | Macro F1 | Fit time |
|---|---:|---:|---:|---:|---:|
| 128 | 1,764 | **2.84%** | 8.06% | **2.48%** | **265.1 s** |
| 224 | 6,084 | 2.32% | **8.10%** | 2.08% | 807.3 s |

### 4.2 Descriptor ablation

At \(128 \times 128\) and \(C=1\), the Combined representation achieved the
best result on every reported validation metric. Combining colour and texture
with HOG improved Top-1 accuracy from 2.84% to 4.80%. The paired improvement
over HOG was \(+1.96\) percentage points (95% CI: \(+1.46\) to \(+2.48\));
McNemar's \(p<1.8\times10^{-14}\).

| Descriptor | Dimension | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 | Fit time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Colour | 57 | 2.16% | 7.70% | 2.58% | 2.16% | 1.62% | 8.5 s |
| LBP | 160 | 2.50% | 7.58% | 2.47% | 2.50% | 1.78% | 19.9 s |
| HOG | 1,764 | 2.84% | 8.06% | 2.72% | 2.84% | 2.48% | 265.1 s |
| Combined | 1,981 | **4.80%** | **12.54%** | **5.13%** | **4.80%** | **4.25%** | 289.1 s |

### 4.3 Regularization ablation

Using the Combined representation, \(C=1\) produced the strongest validation
result. Increasing \(C\) to 10 reduced Top-1 accuracy by 0.90 percentage points
(95% CI: 0.38 to 1.42 percentage points in favour of \(C=1\)); McNemar's
\(p=0.0012\). The final configuration was therefore frozen at \(C=1\).

| \(C\) | Top-1 | Top-5 | Macro F1 | Fit time |
|---:|---:|---:|---:|---:|
| 0.01 | 3.48% | 9.52% | 1.78% | 29.8 s |
| 0.1 | 4.06% | 11.00% | 2.74% | 72.2 s |
| 1 | **4.80%** | **12.54%** | **4.25%** | 289.1 s |
| 10 | 3.90% | 11.78% | 3.89% | 258.4 s |

## 5. Final test result

The frozen \(128 \times 128\), Combined, \(C=1\) model was evaluated once on
the 5,000-image test set. It achieved 4.48% Top-1 accuracy (224/5,000) and
12.16% Top-5 accuracy (608/5,000), with 4.90% macro precision, 4.48% macro
recall, and 3.99% macro F1. Test feature extraction required 24.97 seconds and
prediction required 0.029 seconds for all 5,000 images. The test result was
close to validation performance (4.80% Top-1, 12.54% Top-5, and 4.25% macro
F1), suggesting that validation-based selection transferred consistently to
the held-out split.

| Split | Top-1 | Top-5 | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 4.80% | 12.54% | 5.13% | 4.80% | 4.25% |
| Test | **4.48%** | **12.16%** | **4.90%** | **4.48%** | **3.99%** |

## 6. Error analysis and limitations

Frequent validation errors included visually or contextually related species,
such as *Ceratiola ericoides* being predicted as *Calluna vulgaris* (four
images), *Sequoiadendron giganteum* as *Taxodium mucronatum* (three images),
and *Chaetodon capistratus* as *Parupeneus spilurus* (three images).
Cross-taxon errors also occurred, including cactus-to-snake and
mammal-to-bird predictions.

Qualitative inspection showed that successful cases often contained a large,
clearly visible organism with distinctive colour, texture, or silhouette. In
contrast, many high-scoring failures contained a small or distant subject and
dominant sky, vegetation, branches, or artificial textures. This indicates
that the global handcrafted representation is sensitive to background context
and low-level appearance but lacks the high-level semantic representation
required for fine-grained 500-species recognition. Some incorrect examples
also received larger decision scores than correct examples; these scores are
therefore used only for ranking and are not interpreted as calibrated
probabilities.

Despite its limited accuracy, this method provides an interpretable and
computationally inexpensive traditional baseline. The descriptor ablation
demonstrates that colour, texture, and shape provide complementary
information. Future improvements could use local keypoint aggregation,
foreground localization, spatial pyramids, or learned deep representations to
reduce sensitivity to background and scale.

## 7. Recommended report figures

Use only the figures needed for the final 10-page report:

1. `experiment_workspace/results/formal_validation_comparison.png` for the
   descriptor, resolution, and \(C\) ablations.
2. `experiment_workspace/results/formal_combined_128_c1/top_confusion_submatrix.png`
   for readable validation confusion pairs.
3. `experiment_workspace/results/formal_combined_128_c1/validation_success_failure_cases.png`
   for representative successes and failures.
4. The full 500-class confusion matrix can be placed in supplementary video
   material if page space is limited.

## 8. References to add to the shared BibTeX file

- `vanhorn2021benchmarking`: Grant Van Horn et al., *Benchmarking
  Representation Learning for Natural World Image Collections*, CVPR 2021.
- `ojala2002multiresolution`: Timo Ojala, Matti Pietikäinen, and Topi Mäenpää,
  *Multiresolution Gray-Scale and Rotation Invariant Texture Classification
  with Local Binary Patterns*, IEEE TPAMI, 2002.
- `dalal2005histograms`: Navneet Dalal and Bill Triggs, *Histograms of Oriented
  Gradients for Human Detection*, CVPR 2005.
- `cortes1995support`: Corinna Cortes and Vladimir Vapnik, *Support-Vector
  Networks*, Machine Learning, 1995.
- `pedregosa2011scikit`: Fabian Pedregosa et al., *Scikit-learn: Machine
  Learning in Python*, JMLR, 2011.

## 9. Integration notes for Member D

- Do not report the 20-class pilot as a final result.
- Describe the classifier transparently as a linear hinge-loss/SVM objective
  optimized with SGD, not as kernel SVC.
- Keep test results separate from validation ablations.
- State that model selection used validation only and test was evaluated once.
- The report specification requires the official CVPR LaTeX template,
  `\usepackage[pagenumbers]{cvpr}`, at most 10 main-text pages, and a PDF below
  10 MB.
