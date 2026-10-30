# VIRCON-ERT: Event-aware humanitarian crisis-text classification

**Reproducibility materials** for the study **“VIRCON-ERT: An Event-Aware Triage Transformer for Humanitarian Social Media Classification and Disaster Response”** by S. Mahalakshmi and R. Bagavathi Lakshmi.

- **Public repository:** https://github.com/Mahamangai/VIRCON-ERT-Reproducibility
- **Archived v1.0.0 release:** https://doi.org/10.5281/zenodo.23752635
- **Automated saved-result checks:** https://github.com/Mahamangai/VIRCON-ERT-Reproducibility/actions/workflows/verify-results.yml

## Contents

| Directory | Research materials |
| --- | --- |
| `notebooks/` | Google Colab notebooks for evaluation, transformer experiments, ablations, and audit analysis |
| `scripts/` | Evaluation scripts and saved-output verification |
| `configs/` | Model configurations, seeds, and inference/selection protocols |
| `data_provenance/` | Sample accounting, taxonomy mapping, canonical event mapping, event fold assignments, and input hashes |
| `results/` | Fixed-split and event-disjoint predictions, baseline comparisons, external HumAID results, statistical tests, and audit summaries |
| `environment/` | Captured package and environment information |
| `reviewer/` | Mapping between IJIES reviewer comments and associated outputs |

Large event-disjoint prediction data are stored in compressed `.csv.gz` format, which is read directly by the verification script.

## Verify saved results

With Python and the dependencies specified by `requirements-ci.txt` (or the archived `requirements-verify.txt`), run these commands from the repository root:

```bash
python -m pip install -r requirements-ci.txt
python scripts/verify_saved_results.py
python scripts/check_release_inventory.py
```

The saved-results test recalculates selected classification scores, triage metrics, event-disjoint comparisons, audit agreement, and statistical summaries from archived experiment outputs. The inventory test checks archived file checksums and selected CSV column names. The GitHub Actions workflow runs these checks automatically.

**Scope:** These checks validate the saved results and artifact integrity; they are not a clean-room GPU retraining of the transformer models. Original training notebooks may require authorized source datasets, model downloads, and updates to Google Colab paths or dependencies.

## Model and evaluation

VIRCON-ERT combines a DistilBERT encoder, humanitarian classification and three-way triage heads, an auxiliary request head, cross-event supervised contrastive regularization, and an event-adversarial objective. Training-only objectives are not used during inference.

The repository preserves the difference between the official CrisisBench fixed-split analysis, strict same-corpus event-disjoint evaluation, and external HumAID transfer. The event-disjoint gain over fine-tuned DistilBERT is small and not statistically significant by the archived event-cluster bootstrap analysis. The event-disjoint request class has restricted event coverage; these results do not establish general reliability for unseen disasters.

## Dataset sources

CrisisBench and HumAID source datasets are obtained from their original providers subject to applicable usage requirements. Original input message text is not supplied in this repository. Prepared input hashes and dataset-accounting details are recorded in `data_provenance/`. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Citation

Please cite the relevant manuscript and the archived software release:

Mahalakshmi, S., & Bagavathi Lakshmi, R. (2026). *VIRCON-ERT: An Event-Aware Triage Transformer for Humanitarian Social Media Classification and Disaster Response — Reproducibility Materials* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23752635

See [CITATION.cff](CITATION.cff) for citation metadata and [CITATION_AND_LICENSE.md](CITATION_AND_LICENSE.md) for attribution and reuse information.
