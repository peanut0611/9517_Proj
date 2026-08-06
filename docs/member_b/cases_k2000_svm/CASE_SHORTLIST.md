# Case study shortlist — SIFT-BoVW SVM (K=2000, test)

Source: `/Users/peanut/Code/UNSW/COMP9517/9517_Proj/outputs/b1_ablate_k2000/svm/predictions_test.csv`

## High-confidence correct predictions

| # | true | pred | conf | image_path |
|---|---|---|---:|---|
| 1 | Papilio thoas | Papilio thoas | 0.0108 | `val/02034_Animalia_Arthropoda_Insecta_Lepidoptera_Papilionidae_Papilio_thoas/cf333c0e-b66f-4e1f-9fcb-bc62a6db3328.jpg` |
| 2 | Dennstaedtia punctilobula | Dennstaedtia punctilobula | 0.0091 | `val/09928_Plantae_Tracheophyta_Polypodiopsida_Polypodiales_Dennstaedtiaceae_Dennstaedtia_punctilobula/3c386acd-8a97-4fdd-83f7-47d32ee95d42.jpg` |
| 3 | Clinocardium nuttallii | Clinocardium nuttallii | 0.0086 | `val/05220_Animalia_Mollusca_Bivalvia_Cardiida_Cardiidae_Clinocardium_nuttallii/40ff85a6-5942-4c6e-96c0-3da7ed0354e7.jpg` |
| 4 | Xylobolus frustulatus | Xylobolus frustulatus | 0.0085 | `val/05714_Fungi_Basidiomycota_Agaricomycetes_Russulales_Stereaceae_Xylobolus_frustulatus/837e3a2c-3c9e-4f7c-86f4-a753616470d8.jpg` |
| 5 | Xylobolus frustulatus | Xylobolus frustulatus | 0.0083 | `val/05714_Fungi_Basidiomycota_Agaricomycetes_Russulales_Stereaceae_Xylobolus_frustulatus/14e7200f-e4ee-428f-923d-63513918566b.jpg` |
| 6 | Clinocardium nuttallii | Clinocardium nuttallii | 0.0083 | `val/05220_Animalia_Mollusca_Bivalvia_Cardiida_Cardiidae_Clinocardium_nuttallii/81acb8ae-bf43-4ed1-b2a9-db8fe560a137.jpg` |
| 7 | Torilis japonica | Torilis japonica | 0.0082 | `val/06543_Plantae_Tracheophyta_Magnoliopsida_Apiales_Apiaceae_Torilis_japonica/d47d2d29-1c9c-462e-a004-83161ea631f0.jpg` |
| 8 | Papilio thoas | Papilio thoas | 0.0079 | `val/02034_Animalia_Arthropoda_Insecta_Lepidoptera_Papilionidae_Papilio_thoas/564918d9-41c1-419e-808e-e536483e8e45.jpg` |
| 9 | Euphydryas phaeton | Euphydryas phaeton | 0.0077 | `val/01854_Animalia_Arthropoda_Insecta_Lepidoptera_Nymphalidae_Euphydryas_phaeton/75fb34b9-9ea6-498d-8f79-14d8ce0b66ff.jpg` |
| 10 | Gleichenia dicarpa | Gleichenia dicarpa | 0.0077 | `val/09877_Plantae_Tracheophyta_Polypodiopsida_Gleicheniales_Gleicheniaceae_Gleichenia_dicarpa/bd64ee80-c935-4946-9603-edf074e18be7.jpg` |

## High-confidence errors (model is wrong but confident)

| # | true | pred | conf | image_path |
|---|---|---|---:|---|
| 1 | Hypena bijugalis | Platynota rostrana | 0.0078 | `val/01047_Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Hypena_bijugalis/54a1b518-a7c8-446d-909b-2d2c4ff2fbf7.jpg` |
| 2 | Enallagma geminatum | Clinocardium nuttallii | 0.0065 | `val/02375_Animalia_Arthropoda_Insecta_Odonata_Coenagrionidae_Enallagma_geminatum/c7483ed7-d1f2-4ade-a844-b936bc05aa49.jpg` |
| 3 | Phegopteris hexagonoptera | Dennstaedtia punctilobula | 0.0064 | `val/09990_Plantae_Tracheophyta_Polypodiopsida_Polypodiales_Thelypteridaceae_Phegopteris_hexagonoptera/cdf83d96-d154-4762-8c3b-5dee2cca0885.jpg` |
| 4 | Ponometia semiflava | Clinocardium nuttallii | 0.0062 | `val/01691_Animalia_Arthropoda_Insecta_Lepidoptera_Noctuidae_Ponometia_semiflava/edd6969d-dd5f-4e27-ab45-ff4b103d3b06.jpg` |
| 5 | Urola nivalis | Platynota rostrana | 0.0060 | `val/00949_Animalia_Arthropoda_Insecta_Lepidoptera_Crambidae_Urola_nivalis/635edc80-299a-4619-9105-dd04b210a16b.jpg` |
| 6 | Saturnia pavonia | Henosepilachna vigintioctopunctata | 0.0060 | `val/02166_Animalia_Arthropoda_Insecta_Lepidoptera_Saturniidae_Saturnia_pavonia/c5016f5b-f167-404f-bf9e-61c567ed03b7.jpg` |
| 7 | Chrysopa oculata | Cydia latiferreana | 0.0058 | `val/02297_Animalia_Arthropoda_Insecta_Neuroptera_Chrysopidae_Chrysopa_oculata/e2850df9-896b-475a-91f9-ad8acb21b354.jpg` |
| 8 | Adonis vernalis | Leucanthemum vulgare | 0.0057 | `val/09202_Plantae_Tracheophyta_Magnoliopsida_Ranunculales_Ranunculaceae_Adonis_vernalis/2edb82cb-a985-4381-9bc2-9e5028d13be2.jpg` |
| 9 | Hemiscolopendra marginata | Aquila heliaca | 0.0057 | `val/00157_Animalia_Arthropoda_Chilopoda_Scolopendromorpha_Scolopendridae_Hemiscolopendra_marginata/bd03202d-2aa5-49eb-bc8e-747bac6c8430.jpg` |
| 10 | Chaetodon capistratus | Clinocardium nuttallii | 0.0056 | `val/02812_Animalia_Chordata_Actinopterygii_Perciformes_Chaetodontidae_Chaetodon_capistratus/eddf3297-1daa-4bbf-b331-d483aec9f0d8.jpg` |

## How to use

1. Open paths under `Dataset_v1.0_FROZEN/`.
2. Prefer 3–5 successes and 3–5 failures in the final PDF/video.

