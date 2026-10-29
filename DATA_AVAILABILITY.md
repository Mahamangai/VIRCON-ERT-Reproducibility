# Data and model availability

This repository contains **saved, author-curated research outputs** and preparation/validation metadata. It **does not contain original CrisisBench or HumAID text datasets**, full model checkpoints, or an independently executed clean GPU retraining log.

## Obtain inputs separately

- Obtain CrisisBench and HumAID through their official dataset distribution channels under the providers' terms.
- The training and test files used for prior experiments were prepared/filtered local `.tsv` files, not necessarily the unmodified upstream partitions.
- Compare their SHA-256 hashes to `data_provenance/input_file_hashes.json`.
- Review `data_provenance/complete_sample_accounting.csv`, `data_provenance/event_disjoint_exclusion_detail.csv`, `data_provenance/canonical_event_map_and_counts.csv` and `data_provenance/event_disjoint_fold_assignment.csv`.

## Public-release review required

The 200-message annotation records under `results/manual_audit/` are retained as evidence, but **their release must be approved by the authors and raters**. Redact/remove row-level ratings if permission cannot be established. Source-message text is not included in the curated results; do not add raw text tables without checking permissions and privacy.

## Meaning of reproduction

`python scripts/verify_saved_results.py` recalculates selected metrics from saved model predictions. It does **not** train the neural network from scratch. Original experiments use Google Colab, dependencies, model downloads and paths that need adjustment for a clean rerun.
