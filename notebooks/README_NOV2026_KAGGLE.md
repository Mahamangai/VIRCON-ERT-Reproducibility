# Kaggle notebooks for the November 2026 external benchmark

This directory contains three additional notebooks archived from the author's Kaggle sessions on 21 November 2026. Saved cell outputs and platform-specific metadata were removed in the public copies; the original uploaded notebooks are preserved privately.

## Notebooks and execution sequence

1. **`VIRCON_ERT_Round2_Checkpoint_Grid_Kaggle.ipynb`**: earlier five-fold, seed-based checkpoint-grid experiments. Requires the original historical data, checkpoint-grid trainer, and private state files; **not** a standalone benchmark.
2. **`VIRCON_ERT_Round2_Recovery_And_Independence_Kaggle.ipynb`**: intermediate continuation for recovery of 62 recorded results, 40 detailed run folders, revised trainer, and independent-event preprocessing. Contains embedded research code and summary-result archives, not complete source datasets or model checkpoints. This notebook has cells from different stages and is **not** a proven one-click end-to-end replay.
3. **`IJIES_TurkeySyria_2023_Frozen_Three_Model_Analysis_Kaggle.ipynb`**: the latest completed frozen Turkey–Syria 2023 independent-event evaluation and 3-model comparison (VIRCON-ERT, Crisis CT-M1, and TF-IDF + LinearSVC), followed by post-benchmark bootstrap/error analysis. It requires matching archived training TSV, frozen annotated external-event dataset, and model checkpoint artifacts.

## Reproducibility boundary

Text-free prediction, evaluation, error-analysis, benchmark and checksum outputs live in [`results/external_turkey_syria_2023/`](../results/external_turkey_syria_2023/), with saved-output checks in `scripts/verify_turkey2023_external.py`. This repository's checks verify archived output consistency, **not** full model retraining. This new notebook archive has **not** been clean-run from the public GitHub repository.

The independent-event evaluation uses 600 Turkey–Syria 2023 messages (504 disaster-relevant); five actionability labels marked uncertain are excluded from actionability metrics. Sample counts, model variants, output hashes, and limitations are documented with the saved results.

## Data privacy and licensing

The original message text, licensing-restricted CrisisBench TSVs, private Kaggle data, author credentials, and large model checkpoint binaries are not included in the new notebooks package. The notebooks retrieve secrets from Kaggle at runtime and **do not bundle their values**. Some intermediate notebook cells contain private Kaggle dataset references and embedded historical metadata or metrics, so users should confirm licensing before redistributing or running.

## Publication and versioning

These notebooks are new additions intended for a **future GitHub/Zenodo version**. Earlier published tags (`v1.0.0`, `v1.1.0`) are not altered. A new release may be created only after reviewers check its version tag and confirm the final research artifacts and dataset/license permissions.
