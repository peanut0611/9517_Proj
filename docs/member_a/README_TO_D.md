# Member A -> Member D: Report Material

Please use these files when integrating the final CVPR report.

## Text and configuration

- `member_a_report_draft.md`: Dataset, experimental setup, Member A method,
  ablations, results and limitations.
- `FINAL_MODEL_CONFIG.json`: frozen Member A model and test record.

## Unified group tables

- `VALIDATION_METHOD_COMPARISON.md` / `validation_method_comparison.csv`
- `TEST_METHOD_COMPARISON.md` / `test_method_comparison.csv`

## Figures

- `formal_validation_comparison.png`: Member A validation ablations.
- `top_confusion_submatrix.png`: most frequent validation confusions.
- `validation_success_failure_cases.png`: representative validation cases.

## Member A final result

- Method: 128 x 128 Combined Colour + LBP + HOG, linear SVM objective, C=1.
- Validation: Top-1 4.80%, Top-5 12.54%, Macro F1 4.25%.
- Test: Top-1 4.48%, Top-5 12.16%, Macro F1 3.99%.

Please preserve the timing caveats stated in the comparison files. Traditional
and deep methods did not use identical timing boundaries.
