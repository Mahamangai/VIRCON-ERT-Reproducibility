# Repository access and reproducibility

## Source and release

GitHub repository: https://github.com/Mahamangai/VIRCON-ERT-Reproducibility

Archived v1.0.0 software record: https://doi.org/10.5281/zenodo.23752635

## Download

Use **Code → Download ZIP** on GitHub or clone the repository:

```bash
git clone https://github.com/Mahamangai/VIRCON-ERT-Reproducibility.git
cd VIRCON-ERT-Reproducibility
```

## Run saved-result verification

```bash
python -m pip install -r requirements-ci.txt
python scripts/verify_saved_results.py
python scripts/check_release_inventory.py
```

The GitHub Actions workflow also runs these checks automatically on the designated branch. They verify archived results and checksums, not end-to-end GPU retraining. See `README.md` and `DATA_AVAILABILITY.md` for evaluation scope and source-data information.
