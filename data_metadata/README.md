# Dataset metadata v1.0

Status: **FROZEN**

- Random seed: `9517`
- Classes: `500`
- Train images: `20000` (40 per class)
- Validation images: `5000` (10 per class)
- Test images: `5000` (10 per class)
- Removed category: `9918`
- Replacement category: `705` (Bombus vancouverensis)
- Preserved class index: `495`
- Other 499 class mappings: unchanged
- Train/validation source: official iNaturalist-2021 `train_mini`
- Test source: official iNaturalist-2021 validation split

This release passed Member A's metadata, readability, and SHA256 audits,
followed by an independent audit by Member C on 2026-07-18.

Independent audit results:

- Metadata checksums: PASS
- Dataset structure and readability: PASS
- SHA256 content duplication audit: PASS

Dataset v1.0 is now FROZEN. Do not modify its classes, split membership, labels,
class indices, images, or metadata without creating a documented new version.
