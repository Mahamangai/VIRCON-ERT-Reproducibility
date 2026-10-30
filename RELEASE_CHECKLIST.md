# Repository verification record

This document describes the checks relevant to the publicly archived VIRCON-ERT research materials.

## Automated checks

The GitHub Actions workflow `verify-results.yml` runs the saved-results verification and archived-file integrity checks. It confirms the selected stored metrics, expected file presence, and SHA-256 checksums for the original research-artifact set. The workflow does not retrain the transformer models.

The current GitHub workflow is available at:
https://github.com/Mahamangai/VIRCON-ERT-Reproducibility/actions/workflows/verify-results.yml

## Reproduction boundaries

- The fixed-split, same-corpus event-disjoint, and external HumAID results come from distinct evaluation populations.
- Model selection in the development event-disjoint experiments is disclosed separately from frozen confirmation.
- Original sparse CPU latency and transformer CPU latency measurements were collected in different environments.
- Full clean-environment training reproduction remains separate from the existing saved-results checks.

## Archive record

The v1.0.0 release is available at https://doi.org/10.5281/zenodo.23752635. New research or documentation versions should be traceable to the corresponding version-specific DOI.
