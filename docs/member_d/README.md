# Member D — transfer learning (Yongqi Yang)

Final package synced from OneDrive `Member_D_shared_end/` (2026-08-06).

## Models

| Model | Test Top-1 | Test Top-5 | Macro-F1 |
|---|---:|---:|---:|
| ResNet-18 ImageNet pretrained | 65.10% | 83.34% | 64.42% |
| EfficientNet-B0 ImageNet pretrained | **76.66%** | **90.50%** | **76.20%** |

## Local paths

- Notebook: `notebooks/Member_D.ipynb`
- Metrics / history: `experiment_workspace/D_member_outputs/{resnet18,efficientnet_b0}_pretrained/`
- Report draft: `member_d_report_draft.md`, `D部分_报告.docx`
- Video slides: `video/D_presentation_P8-13.pptx`
- Sync log: `SYNC_FROM_ONEDRIVE.md`

Large artefacts (`best_checkpoint.pth`, `*_scores.npz`, `D_Member.mp4`) stay on OneDrive if not already present locally.
