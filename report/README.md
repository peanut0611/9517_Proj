# COMP9517 Group Report (CVPR LaTeX)

Official deliverable: **PDF ≤ 10 pages body** (references unlimited), **≤ 10 MB**,
using `\usepackage[pagenumbers]{cvpr}`.

## Build

```bash
cd report
latexmk -pdf main.tex
# output: main.pdf
```

Requires a TeX Live / MacTeX install with `pdflatex` + `bibtex`.

## Before Moodle upload

1. Optional: add zIDs next to author names in `main.tex` if desired.
2. Re-read numbers against frozen artefacts if anything was retrained:
   - A/B/C/D unified test table: `docs/member_a/TEST_METHOD_COMPARISON.md`
   - B ablations: `docs/member_b/member_b_results_summary.md`
   - C C0/C1: `docs/member_c/` / C metrics JSON
   - D: `experiment_workspace/D_member_outputs/*/test_metrics.json`
3. Confirm PDF page count ≤ 10 (body) and file size ≤ 10 MB.
4. One group member submits the PDF with the video and code ZIP.

## Authors (role order)

| Role | Name | Focus |
|---|---|---|
| A | Xiangze Xue | Dataset, eval, Colour/LBP/HOG |
| B | Zhigeng Qiao | SIFT-BoVW, repo/ZIP |
| C | Zhiyang Ding | Scratch ResNet-18 |
| D | Yongqi Yang | Pretrained CNNs, report LaTeX |
| E | Xinyan Chen | Grad-CAM / error analysis |

## Source map (who wrote what)

| Report section | Primary sources merged |
|---|---|
| Dataset / protocol / Method A | `docs/member_a/member_a_report_draft.md` |
| Method B / BoVW ablation | `docs/member_b/member_b_report_draft.md` |
| Method C / aug ablation | `docs/member_c/member_c_report_merge.md` |
| Method D | D configs + test metrics |
| Grad-CAM / errors | `docs/member_e/report_notes_E.md` |
| Main comparison table | `docs/member_a/TEST_METHOD_COMPARISON.md` |

Figures live in `figures/`. Large Grad-CAM panels were downscaled for the PDF size budget.

## Spec checklist

- [x] CVPR template + `\usepackage[pagenumbers]{cvpr}`
- [x] Introduction, Literature Review, Methods, Experimental Results, Discussion, Conclusion, References
- [x] Traditional + deep; scratch + pretrained; ≥2 architectures; ablations
- [x] Top-1 / Top-5 / macro metrics + timing + confusion / Grad-CAM discussion
- [x] Author names filled in
- [ ] Final visual pass (figure captions, Overleaf sync if preferred)
