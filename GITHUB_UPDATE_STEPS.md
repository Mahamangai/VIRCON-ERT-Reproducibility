# Upload these additions to the existing GitHub repository

This folder is an **add-on** for your existing public GitHub repository, not a replacement for the archived v1.0.0 release. Do **not** upload the ZIP itself or delete the existing files.

Copy the contained paths to the repository root, preserving the `.github/workflows` directory:
- `.github/workflows/verify-results.yml`
- `requirements-ci.txt`
- `scripts/check_release_inventory.py`
- `notebooks/Verify_VIRCON_ERT_Saved_Results.ipynb`
- `REPRODUCE.md`

`GITHUB_UPDATE_STEPS.md` is optional supporting documentation.

Recommended workflow:
1. Use GitHub Desktop to clone/open `Mahamangai/VIRCON-ERT-Reproducibility` and copy the contents of this add-on folder into the local repository.
2. Review the new files. Commit using `Add automated reproducibility checks` and **Push origin**.
3. On GitHub, select **Actions → Verify VIRCON-ERT saved results** and inspect the latest run. A green check confirms that the two verification scripts passed, **not** that neural-network training was replayed.
4. Open `notebooks/Verify_VIRCON_ERT_Saved_Results.ipynb` in Google Colab to test it yourself.
5. After author review, publish a new GitHub release such as **v1.1.0** and verify Zenodo archives that version. Do not change the original v1.0.0 DOI to pretend new code existed there.

Why new release? The archived v1.0.0 DOI `10.5281/zenodo.23752635` permanently represents the initial software snapshot. New CI and notebook functionality belong in a subsequent snapshot.
