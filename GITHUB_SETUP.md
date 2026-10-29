# Upload VIRCON-ERT to GitHub (beginner guide)

**The GitHub repository should contain the extracted FILES and FOLDERS, not only this ZIP archive.**

1. Sign in at https://github.com/ and choose **New repository**.
2. Name it `VIRCON-ERT-Reproducibility`, description `Code and results for the VIRCON-ERT humanitarian crisis classification study`.
3. Choose **Private** initially, and **do not initialize** it with README, .gitignore or a licence — those files or notes are in this package.
4. Extract the downloaded ZIP onto your computer. Open `VIRCON-ERT-Reproducibility` and review `RELEASE_CHECKLIST.md` and `DATA_AVAILABILITY.md` before publishing.
5. On the GitHub empty-repository page, select **uploading an existing file**, or in an existing repository select **Add file → Upload files**. Drag the **contents** of the extracted `VIRCON-ERT-Reproducibility` directory (folders and individual root files) into the upload area, preserving folder structure.
6. Commit with `Add VIRCON-ERT reproducibility materials`.
7. Confirm that `README.md` is shown on the main repository page and `notebooks`, `scripts`, `data_provenance` and `results` are folders. Check that `scripts/verify_saved_results.py` exists.
8. A reviewer can install `requirements-verify.txt` and run `python scripts/verify_saved_results.py` after cloning/downloading. Large `erht_event_disjoint_predictions.csv.gz` is automatically recognized by the verifier.
9. Check all author permissions, choose and add an appropriate `LICENSE`, and run a clean rerun before changing to **Public**. Do not assert end-to-end reproduction until retraining is successfully validated.
10. After publication approval, create a tagged release such as `v1.0.0`. Connect the public repository to Zenodo, archive a release, obtain a real DOI, and update the manuscript's Data and Code Availability section and reviewer responses.

**GitHub browser constraints (current GitHub docs):** at most 100 files in a single upload and 25 MiB per individual file. This bundle is prepared to fit both limits by shipping one large predictions table as `.csv.gz`. If web upload fails, use GitHub Desktop instead (https://desktop.github.com/).

**No automatic publishing has occurred.**
