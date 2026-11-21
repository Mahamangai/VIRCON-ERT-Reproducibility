# How to reproduce VIRCON-ERT findings

This repository currently supports **two distinct, explicitly bounded workflows**.

## A. Recalculate saved results (automated, no GPU)

Requires Python 3.11 or a compatible environment and normal public internet access to install Python packages.

```bash
git clone https://github.com/Mahamangai/VIRCON-ERT-Reproducibility.git
cd VIRCON-ERT-Reproducibility
python -m pip install -r requirements-ci.txt
python scripts/verify_saved_results.py
python scripts/check_release_inventory.py
python scripts/verify_turkey2023_external.py
```

The first script recalculates selected accuracy, macro-F1, event-fold, annotation agreement, and bootstrap checks from **saved output files**. The second verifies the SHA-256 hashes of the frozen release files and checks some basic package hygiene. **Neither retrains a model.**

### GitHub Actions

Every push or pull request to `main`, as well as a manual invocation, runs `.github/workflows/verify-results.yml`. Look for a **green checkmark** under **Actions → Verify VIRCON-ERT saved results**. CI checks do not imply successful clean training.

## B. Full experiment reproduction (GPU; manual sequence not yet validated)

Original development and final confirmation were performed in Google Colab on a Tesla T4 GPU. GitHub Actions in this repository does **not** run full GPU training. For a training reproduction, a researcher must:

1. Obtain the CrisisBench and HumAID datasets from their respective approved providers. Do **not** commit the original message text to GitHub without checking redistribution restrictions.
2. Reconstruct the study's **prepared/filtered** CrisisBench TSV input files. Their provenance differs from the original publisher partitions. See `data_provenance/complete_sample_accounting.csv` and `input_file_hashes.json`.
3. Verify original input SHA-256 hashes with `data_provenance/input_file_hashes.json` (they describe the *prepared* files, not upstream datasets).
4. Load `data_provenance/canonical_event_map_and_counts.csv`, `event_disjoint_fold_assignment.csv` and the exact triage mapping. Do not generate fresh random folds for a purported exact replay.
5. Start with `notebooks/VIRCON_IJIES_Reviewer_Revision_SELF_CONTAINED_COLAB_SAFE.ipynb` and the companion `scripts/VIRCON_IJIES_Rigorous_Reviewer_Rerun.py` to reproduce sparse/MiniLM baselines.
6. Follow `notebooks/VIRCON_IJIES_Continuation_Full_Strengthening.ipynb` for fine-tuned DistilBERT and external HumAID evaluations, then `notebooks/VIRCON_ERHT_Final_Model.ipynb` for event-aware ablations. Read the final configuration in `configs/vircon_ert_final_confirmation.json`.
7. Record the seed, PyTorch/CUDA/Transformers versions, hardware, file hashes, actual commands, runtime logs and resulting predictions. Distinguish architecture-development results from frozen confirmation results.
8. Recalculate metrics on the newly generated predictions and compare with `results/`. Minor differences across GPU kernels or dependencies may occur even under deterministic controls.

**Limitations:** Original Colab notebooks include author-specific Google Drive paths and prior-artifact ZIP dependencies. They need to be parameterized and clean-run-tested before this project may accurately claim one-command, end-to-end training reproducibility. `environment/reviewer_rerun_pip_freeze.txt` is from an earlier experiment environment, not necessarily the final GPU run.

## C. One-click verification in Colab

Open `notebooks/Verify_VIRCON_ERT_Saved_Results.ipynb` in Google Colab and run all cells. This checks saved metrics without requiring any prepared CrisisBench TSV files, private Google Drive access, or a GPU.

## D. November 2026 Turkey–Syria external evaluation

Saved-result checks and outputs: [`results/external_turkey_syria_2023/README.md`](results/external_turkey_syria_2023/README.md). The [Kaggle notebook guide](notebooks/README_NOV2026_KAGGLE.md) identifies the checkpoint-grid, recovery, and frozen three-model comparison notebooks and their order. Those notebooks contain sanitized sources with saved cell outputs removed. Exact inference/training requires separately obtained checksum-matched CrisisBench inputs, the frozen 600-message evaluation dataset, and the original VIRCON-ERT and Crisis CT-M1 checkpoint files. These restricted inputs and checkpoint binaries are not in the public archive. A complete fresh GPU replay has not been verified.

## Scholarly versioning

The current archived external-validation release `v1.2.0` is https://doi.org/10.5281/zenodo.24419660 (21 November 2026). Older releases are `v1.1.0` (https://doi.org/10.5281/zenodo.23804759) and `v1.0.0` (https://doi.org/10.5281/zenodo.23752635). The concept DOI https://doi.org/10.5281/zenodo.23752634 represents all versions. These README/citation documentation corrections are made to the current `main` branch **after** the archived v1.2.0 release; they do not rewrite the immutable v1.2.0 Zenodo ZIP or its Git tag. Use the version-specific DOI for exact citation.
