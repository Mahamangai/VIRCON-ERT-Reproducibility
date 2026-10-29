# Pre-publication author checklist

- [ ] Confirm institutional/project ownership and permission to publish analysis code, figures, label mappings and reviewer attachments.
- [ ] Confirm publication of the **200-message audit labels** is permitted under the annotator agreement and privacy/data-protection terms; remove row-level annotations if not.
- [ ] Check all four cleaned notebooks and Python source for passwords, tokens, account paths, personal names, or hard-coded third-party secrets (static scanning is not a security guarantee).
- [ ] Verify original prepared CrisisBench train/test TSV SHA-256 hashes and clarify upstream-vs-filtered dataset count differences in manuscript Methods.
- [ ] Confirm no raw post text or restricted tweet/message content is redistributed. **Input TSVs and raw-text exclusion tables are intentionally excluded.**
- [ ] Confirm that the sample exclusion code, event alias map and fold IDs recreate the same 67,243-row population from lawful source inputs.
- [ ] Confirm that the saved ERHT no-hierarchy event-disjoint result is described as *development/post-selection evidence*, and that the final fixed-split and HumAID results reflect the frozen confirmation.
- [ ] Confirm manual two-rater audit provenance and the final 200 annotations; check if anonymised ratings may be shared.
- [ ] Check 2026-era dependency availability in a fresh Colab runtime; the saved pip freeze is an earlier run, not evidence of exact latest GPU environment.
- [ ] Run `python scripts/verify_saved_results.py` locally and record any failure.
- [ ] Perform a **clean end-to-end training replay** from scratch if claiming fully executable independent reproduction; record input files, seeds, package lock and GPU type.
- [ ] Decide an appropriate software licence and dataset/annotation licence; none is asserted here.
- [ ] Review the Google Drive author's permissions and choose which artifacts to make public.
- [ ] Publish a versioned GitHub release, then register it with Zenodo; replace placeholders in manuscript only after verified URLs/DOIs exist.
- [ ] Update reviewer response letter to reference the **published paths and release version**, rather than asserting public availability prematurely.
