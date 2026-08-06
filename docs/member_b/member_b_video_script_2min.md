# Member B — Template slides + ~1:30 script

**Template:** `docs/group/group_presentation_template.pptx`  
**Name:** Zhigeng Qiao · SIFT + BoVW + Linear SVM  
**Your slides:** **4–5** (hard cap **2:00**; aim **≈1:30**)

---

## Which slides are yours?

| Slide | Title (template) | Your job |
|---:|---|---|
| 3 | Common protocol / method map | Shared overview — **do not own**; A/intro may mention SIFT-BoVW briefly |
| **4** | Traditional pipeline | **Your main method slide** |
| **5** | Traditional results | **Your results + takeaway** |
| 12 | Final comparison table | Group slide — B row filled by whoever integrates; you may cite numbers only if asked |

Demo of success/failure images: show **on slide 5** (extra images) or a brief screen share between 5 → 6; leave full Grad-CAM software demo to **E**.

---

## What to change on the PPT

### Slide 4 — replace all `[brackets]`

**Keep title or use:**  
`Local SIFT keypoints → BoVW → linear SVM`

**Pipeline (already on slide — keep):**  
IMAGE → SIFT → VISUAL VOCABULARY → BoVW VECTOR → SVM  
*(Drop “/ RF” from the final arrow if you only present SVM as primary.)*

**Design choices — fill in:**
- Vocabulary sizes: **K ∈ {500, 1000, 2000}**; select **K = 2000** by val macro-F1
- Descriptor / sampling: grayscale SIFT, `max_side=512`; k-means on **train only**
- Classifier: **LinearSVC** primary (`C=1`, `dual=false`); RF was much weaker → not the headline
- Normalisation: **TF–IDF + L2**

**Why this baseline matters — fill in:**
- Local keypoints capture repeatable texture/edge patterns (vs global colour/HOG)
- Strength: interpretable classical baseline under the **same frozen split**
- Weakness: orderless histograms struggle at **500-way** fine-grained recognition

### Slide 5 — replace placeholders with shared-evaluator numbers

**Title (use as-is or):**  
`Handcrafted BoVW beats chance, but is far from CNN accuracy`

**Big numbers:**
- **8.86%** Held-out Top-1  
- **7.71%** Held-out macro-F1  

*(Also put Top-5 **17.80%** in a small line if space.)*

**Chart:** one bar chart — val (or test) Top-1 / macro-F1 for **K = 500 / 1000 / 2000** (small gaps). Label split clearly.

**One-sentence takeaway:**  
`Above ~0.2% chance, but orderless BoVW cannot match deep features — CNNs next.`

**Optional:** two thumbnails — success *Papilio thoas* + one high-conf error from `docs/member_b/cases_k2000_svm/CASE_SHORTLIST.md`.

---

## Timing (1:30)

| Time | Slide | Content |
|---:|---:|---|
| 0:00–0:10 | 4 | Name + role |
| 0:10–0:45 | 4 | Pipeline + design choices |
| 0:45–1:15 | 5 | Test metrics + K ablation |
| 1:15–1:25 | 5 / demo | 1 success + 1 failure |
| 1:25–1:30 | 5 | Takeaway → hand to C |

---

## Spoken script (English) — ≈1:30 / ~210 words

### [Slide 4 · 0:00–0:45]

Hi, my name is **Zhigeng Qiao**.  
I built our **traditional baseline**: SIFT with Bag-of-Visual-Words and a **linear SVM**.

On the frozen 500-class split, I extract **SIFT** descriptors, learn a visual vocabulary **on training images only**, encode each image as a **BoVW** histogram with **TF–IDF and L2**, then classify with **LinearSVC**.  
Unlike global colour or HOG, this targets **local** repeatable patterns.  
I compared vocabulary sizes **K = 500, 1000, and 2000**, and selected **K = 2000** by validation macro-F1.

### [Slide 5 · 0:45–1:25]

On the held-out test set the shared evaluator gives about **8.9% Top-1**, **17.8% Top-5**, and **7.7% macro-F1** — well above **0.2%** chance for 500 classes, but far below our CNNs.  
Ablating K only moves the numbers slightly, so vocabulary size is not the main bottleneck.

Here is a **correct** case where distinctive local texture helps — for example the butterfly *Papilio thoas*.  
And a **confident failure**, where similar insects or cluttered backgrounds fool an orderless histogram.

### [Close · 1:25–1:30]

So BoVW is a useful classical baseline, but for fine-grained recognition the heavy lifting moves to deep models.  
Thanks — next, scratch ResNet.

---

## 中文提示卡（录英文时看）

- 开场报全名；摄像头开着  
- **第 4 页**：SIFT → 词袋(train only) → TF–IDF → Linear SVM；K=2000  
- **第 5 页**：8.9% / 17.8% / 7.7%；K 消融变化小；1 对 + 1 错  
- 总时长 **≈1:30**，不超过 **2:00**  
- 完整软件演示交给 E  

## Rehearsal

1. Timer once with slides 4→5.  
2. Over **1:40** → cut “Unlike global…” and show only one demo image.  
3. Under **1:15** → add one failure reason (“similar taxa / background”).
