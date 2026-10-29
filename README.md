# VIRCON-ERT: event-aware humanitarian crisis-text classification

**Research reproducibility candidate — not yet approved for public release.**

This repository organizes the experiment code and saved results supporting the manuscript **“VIRCON-ERT: An Event-Aware Triage Transformer for Humanitarian Social Media Classification and Disaster Response”** (S. Mahalakshmi and R. Bagavathi Lakshmi).

## Start here

- **New to GitHub?** Follow [`GITHUB_SETUP.md`](GITHUB_SETUP.md).
- **To verify saved results:** run `pip install -r requirements-verify.txt`, then `python scripts/verify_saved_results.py`. This performs **re-scoring only**, not model training.
- **Reproduction constraints:** [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
- **Before public release:** [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
- **For IJIES review:** [`reviewer/IJIES_REVIEWER_EVIDENCE_MAP.csv`](reviewer/IJIES_REVIEWER_EVIDENCE_MAP.csv).

### Directory map

| Folder | Contents |
| --- | --- |
| `notebooks/` | Cleaned Colab notebooks (human audit, transformer baseline, ERHT ablation) |
| `scripts/` | Reviewer rerun source and saved-prediction verifier |
| `configs/` | Model and evaluation configurations |
| `data_provenance/` | Counts, label taxonomy, event mapping, leakage-safe fold assignment |
| `results/` | Saved predictions, baselines, ablations, HumAID, bootstrap and audit |
| `environment/` | Saved research environment snapshots |
| `reviewer/` | Reviewer-comment evidence crosswalk |

**Large-file note:** one 35 MB CSV is stored as `results/architecture_ablation/erht_event_disjoint_predictions.csv.gz` to permit GitHub browser upload. The verification script reads this compressed file automatically.

---

## Detailed provenance and reviewer notes

**Status:** curated research evidence, prepared for author review. **Not yet published or independently rerun from a clean environment.**

This package relates to the manuscript *VIRCON-ERT: An Event-Aware Triage Transformer for Humanitarian Social Media Classification and Disaster Response*. It consolidates source code and verified exported result artifacts from the author's Google Colab/Google Drive project plus additional reviewer-facing provenance tables.

## Important distinctions

- `VIRCON-ERT` is the final **DistilBERT-based** architecture. `ERHT` is the **development-stage ablation** naming retained by the experiment files. The no-hierarchy ablation in `architecture_ablation` is the selected event-disjoint candidate. Do not treat event-disjoint selection results as an independent holdout.
- Confirmed fixed-split results and externally prepared HumAID results are different populations; do not combine their metrics.
- Original sparse CPU timings and final transformer timings were measured in different environments. They do not establish a matched-hardware Pareto optimum.
- Full model checkpoints and original data are **not redistributed** in this ZIP. Source Colab notebooks and exact prediction tables allow result re-scoring, but do **not** guarantee end-to-end retraining without dependency and path adjustments.
- Some original Colab notebooks rely on Google Drive paths, cached data, or external pretrained-model downloads. Outputs and execution counts have intentionally been cleared to avoid accidentally republishing message text or credentials in display outputs. Check and adapt code paths before a clean rerun.

## Folder contents

- `notebooks/` — sanitized code-only Colab notebook copies (output cells cleared); includes baseline, transformer continuation, ERHT development and manual audit.
- `scripts/` — original reviewer-rerun Python source plus a small post-export verification script.
- `data_provenance/` — sample flow, event alias map, exact triage mapping, fold assignments, input-source hashes, HumAID overlap and sample evidence. **No input social-media message text.**
- `configs/` and `environment/` — captured run parameters, protocol notes, environment records and package-freeze from an earlier reviewer rerun (not necessarily the final GPU environment).
- `results/` — model outputs, ablation/bootstrap results, text-free predictions, deployment summaries and manual audit aggregates.
- `reviewer/` — reviewer-to-artifact evidence crosswalk.
- `RELEASE_CHECKLIST.md` — essential steps before posting this package publicly.
- `SHA256SUMS.txt` and `FILE_MANIFEST.csv` — checksums for the included files.

## Source datasets and input provenance

The original author's inputs are `crisis_consolidated_humanitarian_filtered_lang_en_train.tsv` and `...test.tsv` under their Colab Drive project; these files are deliberately **not included**. The saved SHA-256 hashes can be found at `data_provenance/input_file_hashes.json`. Dataset authors and redistribution licences must be checked before sharing input message text; obtain CrisisBench and HumAID from their respective authorised sources.

The saved `complete_sample_accounting.csv` documents an input with 61,164 training and 17,356 test records (78,520 prepared records). The same-corpus event-filtered evaluation contains 67,243 records after 11,269 generic/unresolved-event exclusions and eight further duplicate/conflicting-text exclusions. The 67,243 saved event-fold rows cover 50 canonical events across 5 folds. `canonical_event_map_and_counts.csv` preserves 59 original event alias keys mapped to those 50 events. Verify the origin of the already-filtered CrisisBench input files before describing these as the unmodified *original publisher* partitions; published upstream counts may differ.

## Recalculate saved numerical results

Create an environment with `pandas` and `scikit-learn` installed and run:

```bash
python scripts/verify_saved_results.py
```

This script checks the 11-class fixed-split accuracy/F1, event-disjoint F1, request and triage metrics, audit agreement and saved bootstrap summary **from existing predictions**. It does not retrain the models or reproduce Colab GPU conditions. It should run from the extracted package root or any working directory.

## Suggested code-reproduction order (requires original data and model downloads)

1. Read `configs/` and `data_provenance/` and verify input file hashes against the author's prepared TSVs.
2. Review and adapt `notebooks/VIRCON_IJIES_Reviewer_Revision_SELF_CONTAINED_COLAB_SAFE.ipynb` and `scripts/VIRCON_IJIES_Rigorous_Reviewer_Rerun.py` for sparse/MiniLM baselines and exact fold reconstruction.
3. Examine `notebooks/VIRCON_IJIES_Continuation_Full_Strengthening.ipynb` for the DistilBERT and HumAID continuation, then `notebooks/VIRCON_ERHT_Final_Model.ipynb` for event-aware ablations and the final configuration in `configs/vircon_ert_final_confirmation.json`.
4. Review `notebooks/Human_Audit_Analysis.ipynb` for audit agreement calculations. The anonymized rating table is in `results/manual_audit/`.
5. Refit models, save separate validation and held-out predictions, then compare with the frozen tables in `results/`. Recording fresh environment hashes and dependency versions is recommended.

No automatic clean-environment full-training verification has been performed for this package. Code remains provenance-preserving and has **not** been rewritten into a fully automated one-command training pipeline.

## Public release conditions

Do not publish until the authors review `RELEASE_CHECKLIST.md`, decide a software licence, approve release of annotator labels, and confirm data-source permissions. The external predictions in this package have their `raw_text` and `clean_text` columns removed. Exact URLs, personal email addresses, and unrestricted raw source text are not included in intentionally curated CSV result files; inspect all code/configurations before publication.
