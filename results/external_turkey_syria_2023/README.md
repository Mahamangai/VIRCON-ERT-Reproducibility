# Turkey–Syria earthquake 2023: independent-event external evaluation

This directory is an **incremental reproducibility supplement** for the public `VIRCON-ERT-Reproducibility` project. It preserves output from a frozen, independently annotated 600-message Turkey–Syria earthquake 2023 test set. It is **not** a replacement for the original CrisisBench fixed-split, event-disjoint, or HumAID experiments.

## Layout

- `benchmark/`: 25 text-free prediction, confusion-matrix, per-class, comparison, inference timing, bootstrap and provenance files. One baseline is `Crisis_CT_M1` (crisis-domain RoBERTa model); one is TF-IDF + class-balanced LinearSVC.
- `analysis/`: 8 text-free supplementary paired-bootstrap, class-error, request-alert and annotation-audit files.
- `NEW_FILES_SHA256SUMS.txt`: SHA-256 integrity list for those 33 archived outputs.
- `../../scripts/verify_turkey2023_external.py`: output integrity and consistency verifier.
- `../../scripts/IJIES_Cell11_Post_Benchmark_Analysis.py`: supplementary post-benchmark analysis source.

## Verification (CPU only)

From repository root, with Python and the existing `requirements-ci.txt` (including pandas, numpy, scikit-learn):

```bash
python -m pip install -r requirements-ci.txt
python scripts/verify_turkey2023_external.py
```

The script checks stored-file hashes and recomputes metrics from the confusion matrices; it **does not reproduce the original model training, original labels, or individual bootstrap draws**.

## External evaluation summary

| Model | All 600: 11-class macro-F1 | Relevant 504: 11-class macro-F1 | Relevant 504: accuracy | Relevant subset request precision / recall |
|---|---:|---:|---:|---:|
| VIRCON-ERT (frozen full) | 0.3155 | 0.3294 | 0.5119 | 0.0526 / 0.7500 |
| CrisisTransformer CT-M1 | 0.3033 | 0.3138 | 0.4722 | 0.0382 / 0.7500 |
| TF-IDF + LinearSVC | 0.2899 | 0.2960 | 0.4206 | 0.0345 / 0.3750 |

The triage evaluation excludes five instances explicitly marked `uncertain`: n=595 across all messages; n=499 on the relevant subset. The author-side confirmation audit found five confirmed review rows, **but those five actionability values remain uncertain**. The 11-class evaluation includes all messages.

Paired 2,000-message-resample bootstrap Macro-F1 difference, VIRCON-ERT minus comparator (95% CI):

| Population | Comparator | Difference | 95% CI |
|---|---|---:|---|
| 600 | Crisis CT-M1 | +0.0122 | [-0.0175, +0.0421] |
| 600 | LinearSVC | +0.0257 | [-0.0098, +0.0599] |
| Relevant 504 | Crisis CT-M1 | +0.0155 | [-0.0209, +0.0503] |
| Relevant 504 | LinearSVC | +0.0334 | [-0.0087, +0.0745] |

These **message-level bootstrap intervals all contain zero**, so the observed Macro-F1 advantages cannot be presented as statistically established. They also do not quantify variation across future disasters.

## Resource benchmark

The two transformer models were profiled using batch size 32 with a Tesla T4 GPU, including tokenization and transfers but excluding model loading. VIRCON-ERT: 310.14 messages/s, 405.03 MiB peak allocated GPU memory, 66.48 M parameters; CT-M1: 167.22 messages/s, 666.27 MiB, 135.02 M parameters. The TF-IDF + SVC model ran on CPU in the same Kaggle session and is **not a GPU-to-GPU comparator**.

## Exact-input provenance and incomplete artifacts

`benchmark/External_Crisis_Benchmark_Manifest.json` records the SHA256 values for the previously frozen 600-message evaluation CSV, prepared historical CrisisBench TSV, full VIRCON checkpoint and crisis checkpoint. Original message text, original prepared input files and trained checkpoint bytes are **not included** in this supplement. The crisis transformer is trained for two epochs on the original historical train set, without training on the Turkey–Syria external labels. The `Crisis_CT_M1` and SVC three-way triage predictions are taxonomy-derived, while VIRCON-ERT uses its direct trained triage head.

For an exact training/inference rerun, retain or provide access to:

1. Frozen 600-row evaluation CSV matching `test_dataset_SHA256` and the five-row confirmed annotation file, where access and redistribution permissions permit.
2. Prepared CrisisBench training TSV matching `original_train_SHA256`.
3. The original VIRCON-ERT checkpoint matching `full_checkpoint_SHA256`.
4. The saved CT-M1 checkpoint matching `crisis_checkpoint_SHA256` (kept outside the small results ZIP), plus tokenizer and training manifest.
5. The complete executed Kaggle notebook that produced this benchmark, including its dataset mounting instructions, seed and environment versions.

All identifiers in the text-free outputs are pseudonymous but are still derived research records. Review dataset, platform and annotation licensing before publishing. Do not publish message text or private paths/tokens without authorization.

## Interpretive limitations

Only one independent disaster was assessed; the request-positive support is n=8 in the relevant subset, with 108 false request alerts for VIRCON-ERT. This is evidence for a **human-supervised research workflow**, not autonomous emergency dispatch. The new comparative evidence must be linked in the revised **VIRCON-ERT** manuscript and must not be silently attributed to an older, sparse-only **VIRCON-Classify** manuscript.
