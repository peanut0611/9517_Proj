# Sync note — Member D final package

Source: OneDrive `COMP9517_GroupProject/Member_D_shared_end/`  
Synced: 2026-08-06

## Important metric update

EfficientNet-B0 **test Top-1 changed** vs earlier handoff:

| Model | Previous (local) | New (`Member_D_shared_end`) |
|---|---:|---:|
| ResNet-18 pretrained Top-1 | 65.10% | **65.10%** (same) |
| ResNet-18 Top-5 | 83.34% | **83.34%** |
| EfficientNet-B0 Top-1 | 76.36% | **76.66%** |
| EfficientNet-B0 Top-5 | 90.16% | **90.50%** |
| EfficientNet-B0 macro-F1 | 75.97% | **76.20%** |

## Synced locally

- `notebooks/Member_D.ipynb`
- `docs/member_d/` — report docx/md, figures, video pptx
- `experiment_workspace/D_member_outputs/{resnet18,efficientnet_b0}_pretrained/` — metrics/config/csv/history (not `.pth` / `.npz`)
- `experiment_workspace/C_member_outputs/predictions/` — analysis exports mirrored from D package

## Not copied into git tree (too large)

- `best_checkpoint.pth`, `*_scores.npz` (may already exist locally from earlier runs)
- `video/D_Member.mp4` (~78 MB) — remains on OneDrive only

## Report / comparison docs

`report/main.tex` and `docs/member_a/*METHOD_COMPARISON*` updated to EffNet **76.66%** / **90.50%** (val Top-1 **75.96%**).
