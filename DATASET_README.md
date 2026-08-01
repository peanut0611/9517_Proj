# COMP9517 Dataset v1.0 - FROZEN

## 1. Dataset status

Current status: **FROZEN**

This repaired release passed Member A's metadata, image readability, path,
image-ID, and SHA256 content audits. Member C then independently audited the
shared release on 2026-07-18 and confirmed:

```text
Metadata checksums: PASS
Dataset audit: PASS
SHA256 audit: PASS
```

The release is therefore frozen as `Dataset v1.0`.

Once frozen, no group member may regenerate the split, change the selected
classes, modify `class_index`, move images between splits, or replace images
without group agreement and a documented version change.

## 2. Dataset overview

This subset was created from the official iNaturalist-2021 dataset for the
COMP9517 group project.

| Item | Value |
|---|---:|
| Random seed | `9517` |
| Number of classes | `500` |
| Training images per class | `40` |
| Validation images per class | `10` |
| Test images per class | `10` |
| Total training images | `20,000` |
| Total validation images | `5,000` |
| Total test images | `5,000` |
| Total images | `30,000` |

- Training and validation images come from the official iNaturalist-2021
  `train_mini` split.
- Test images come only from the official iNaturalist-2021 validation split.
- Class indices are fixed in the contiguous range `0-499`.
- Training, validation, and test records are disjoint by path and image ID.
- A SHA256 audit found no remaining duplicate image content within or across
  splits.

## 3. Documented category repair

The initial dataset audit found identical image content across the training and
held-out test splits for category `9918`.

The repaired version:

- removed category `9918` - *Lomaria discolor*;
- added category `705` - *Bombus vancouverensis*;
- preserved `class_index 495`;
- left all other 499 category-to-index mappings unchanged;
- selected category 705 deterministically from the previously unselected
  eligible categories using seed `9517`;
- retained exactly 40 training, 10 validation, and 10 test images per class.

The removed category images are not included in this shared release.

## 4. Shared folder contents

The SharePoint/OneDrive project folder must contain:

```text
COMP9517_GroupProject/
├── DATASET_README.md
├── data_metadata/
│   ├── README.md
│   ├── audit_report.json
│   ├── class_to_idx.json
│   ├── dataset_summary.json
│   ├── independent_audit_record.json
│   ├── metadata_checksums.sha256
│   ├── selected_classes.json
│   ├── sha256_audit_report.txt
│   ├── taxonomy.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── Dataset_v1.0_FROZEN/
│   ├── train_mini.zip
│   ├── val.zip
│   └── archive_checksums.sha256
└── src/
    └── data/
        ├── audit_splits.py
        ├── audit_sha256.py
        ├── create_splits.py
        ├── extract_subset.py
        ├── prepare_category_repair.py
        └── repair_category_dry_run.py
```

The shared `src/data/` directory should not contain `__pycache__`. The ZIP
archives should not contain `.DS_Store` or `__MACOSX` entries.

## 5. Required local directory structure

Download `train_mini.zip` and `val.zip` from `Dataset_v1.0_FROZEN/`. After
extracting them, each member must have:

```text
<dataset_root>/
├── train_mini/
│   └── <500 selected class directories and 25,000 images>
└── val/
    └── <500 selected class directories and 5,000 images>
```

The image paths recorded in `train.csv`, `val.csv`, and `test.csv` are relative
to `<dataset_root>`. Each member may use a different absolute dataset location,
but must not edit the relative paths in the shared CSV files.

Important: the physical `train_mini/` directory contains both the training and
validation images. Always use `train.csv` and `val.csv` to distinguish them. Do
not load the entire `train_mini/` directory as a single training set.

The physical `val/` directory contains the official iNaturalist validation
images used as the held-out project test set. Always load them through
`test.csv`.

## 6. Metadata files

- `selected_classes.json`: the final 500 selected categories, including category
  705 and excluding category 9918.
- `class_to_idx.json`: the fixed mapping from original category IDs to class
  indices.
- `train.csv`: 20,000 training records, 40 per class.
- `val.csv`: 5,000 validation records, 10 per class.
- `test.csv`: 5,000 held-out test records, 10 per class.
- `taxonomy.csv`: taxonomy information for all final selected classes.
- `dataset_summary.json`: version, status, counts, sources, and repair record.
- `audit_report.json`: latest structural and image-readability audit result.
- `sha256_audit_report.txt`: latest image-content duplication audit result.
- `metadata_checksums.sha256`: SHA256 checksums for the metadata release.

The CSV files include the image path, image ID, original category ID, class
index, taxonomy fields, and split name.

## 7. Mandatory loading and experiment rules

1. Fit model parameters using only `data_metadata/train.csv`.
2. Use `data_metadata/val.csv` for hyperparameter tuning, model selection, and
   ablation decisions.
3. Use `data_metadata/test.csv` only for final evaluation.
4. Do not use the test set for tuning or model selection.
5. Do not recreate, reshuffle, or randomly re-split the shared data.
6. All methods must use the shared `class_to_idx.json` mapping.
7. Method-specific preprocessing and augmentation are allowed, but the
   underlying split membership and labels must remain unchanged.
8. Record the exact dataset version and random seed for every experiment.

## 8. Independent audit record and verification

Member C independently audited the files on 2026-07-18. The metadata checksum,
dataset structure/readability, and SHA256 content audits all passed. The result
is recorded in `data_metadata/independent_audit_record.json`.

Any group member can verify a downloaded copy using the following commands.

From the `COMP9517_GroupProject` directory, first verify the metadata checksums:

```bash
cd data_metadata
shasum -a 256 -c metadata_checksums.sha256
cd ..
```

Then run the structural and image-readability audit:

```bash
python3 src/data/audit_splits.py \
  --metadata-dir data_metadata \
  --data-root <dataset_root>
```

Then run the image-content SHA256 audit:

```bash
python3 src/data/audit_sha256.py \
  --metadata-dir data_metadata \
  --data-root <dataset_root>
```

Replace `<dataset_root>` with the directory containing the extracted
`train_mini/` and `val/` folders.

The two image archives can also be verified from the project directory:

```bash
cd Dataset_v1.0_FROZEN
shasum -a 256 -c archive_checksums.sha256
cd ..
```

Required results:

```text
All metadata checksums: OK
Image archive checksums: OK
Dataset audit: PASS
SHA256 AUDIT PASS
```

Recorded independent audit:

```text
Audited by: Member C
Audit date: 2026-07-18
Metadata checksum result: PASS
Dataset audit result: PASS
SHA256 audit result: PASS
```

This release is `Dataset v1.0 - FROZEN`.

## 9. Sharing and permissions

- Keep previous unverified releases separate and clearly label them
  `RETIRED_DO_NOT_USE`.
- Do not mix image archives from one version with metadata from another version.
- Do not rename `Dataset_v1.0_FROZEN`, `data_metadata`, or the files inside them
  after the final archive and metadata checksums have been published.
- Other members should normally receive view/download access.
- Keep the frozen shared release read-only for normal use.
- Do not commit large image archives, extracted images, trained models, or model
  checkpoints to the Git repository.

## 10. Report and reproduction information

The written report must document:

- dataset version `v1.0`;
- final list of 500 species;
- random seed `9517`;
- 40/10/10 images per class;
- official source of each split;
- category replacement `9918 -> 705`;
- preservation of `class_index 495`;
- that the other 499 mappings remained unchanged;
- that cross-split SHA256 auditing found no remaining duplicate image content.

The scripts used to create, repair, extract, and audit the subset are included in
the shared `src/data/` directory.

Do not silently fix a suspected dataset problem. Report it to the group first so
that any necessary change can be documented and released under a new version.
