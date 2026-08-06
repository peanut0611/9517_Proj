# Member E — Grad-CAM Deliverables

Synced from OneDrive `Member_E_deliverable/E_GradCAM_deliverables/` (2026-08-05).

## Layout in this repo

| Path | Contents |
|---|---|
| [`../../notebooks/Member_E.ipynb`](../../notebooks/Member_E.ipynb) | Main Grad-CAM analysis notebook |
| `summaries/` | Case / comparison CSVs |
| `figures_c1/` | C1 Scratch ResNet-18 Grad-CAM panels |
| `figures_scratch_vs_pretrained/` | C1 vs D ResNet-18 side-by-side |
| `figures_three_model_comparison/` | C1 + D ResNet-18 + D EfficientNet-B0 |
| `notes/` | Report notes, D configs/metrics, case-selection metadata |
| `report_notes_E.md` | Prose notes for the group report |

Canonical location in this repo is `docs/member_e/` (OneDrive packaging copies are no longer mirrored under `handoffs/`).

## Method summary

- **C1:** Scratch ResNet-18 (500 classes), checkpoint by val macro-F1; target layer `model.layer4[-1]`
- **D ResNet-18 pretrained:** target layer `model.layer4[-1]`
- **D EfficientNet-B0 pretrained:** target layer `model.features[-1]`
- Eval transform: Resize(256) → CenterCrop(224) → ImageNet normalize
- Grad-CAM target class: each model’s own predicted class
- Case groups: high-confidence correct, high-confidence error, top confusion pairs

## Submission note

- Notebook + small CSV/JSON → suitable for the ≤40MB code ZIP  
- PNG figures → for **report / video**; omit large figure folders from the code ZIP if size is tight  
- Do **not** include datasets or checkpoints

## Reference

Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV 2017.
