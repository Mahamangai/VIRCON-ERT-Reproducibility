# Data and code availability

The repository provides the research code, configuration records, sample-accounting files, event mappings and fold assignments, saved model predictions, statistical outputs, and annotation summaries supporting the VIRCON-ERT study.

**Repository:** https://github.com/Mahamangai/VIRCON-ERT-Reproducibility

**Archived software (all versions):** https://doi.org/10.5281/zenodo.23752634

## Dataset inputs

CrisisBench and HumAID source datasets are distributed by their original providers and are not included as full raw-message datasets here. The prepared experimental train/test files were filtered inputs rather than necessarily the unmodified publisher partitions. Input hashes are supplied under `data_provenance/` to support matching legally obtained inputs with the experiment records.

The released results include text-free prediction tables and manual-audit result files. Rights to underlying social-media messages and any annotations remain subject to their source conditions.

## Reproducibility scope

`python scripts/verify_saved_results.py` recalculates selected reported numerical metrics from the archived predictions. The continuous-integration workflow also validates the tracked artifact checksums and selected CSV headers. These checks do not constitute independent complete retraining of the neural models from scratch. The notebooks document the historical Colab experiments and may require local data, pretrained-model downloads, and environment updates for a clean retraining attempt.
