#!/usr/bin/env python3
"""Check v1.0.0's archived artifact hashes and basic public-file hygiene.

This does NOT prove that GPU training is reproducible. It validates frozen
research artifacts and flags some common accidental data disclosures.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import csv
import re

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = [
    'README.md',
    'scripts/verify_saved_results.py',
    'configs/vircon_ert_final_confirmation.json',
    'data_provenance/complete_sample_accounting.csv',
    'data_provenance/canonical_event_map_and_counts.csv',
    'data_provenance/event_disjoint_fold_assignment.csv',
    'results/architecture_ablation/erht_event_disjoint_predictions.csv.gz',
    'results/vircon_ert_confirmation/vircon_ert_fixed_predictions.csv',
    'results/manual_audit/triage_manual_audit_FINAL_frozen_summary.csv',
    'results/distilbert_and_external/distilbert_event_disjoint_oof_predictions.csv',
]

missing = [p for p in EXPECTED if not (ROOT / p).is_file()]
if missing:
    raise SystemExit('Missing key research file(s): ' + ', '.join(missing))

sums = ROOT / 'SHA256SUMS.txt'
if not sums.is_file():
    raise SystemExit('The original SHA256SUMS.txt is absent.')

n = 0
for line in sums.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    try:
        expected_hash, rel = line.split('  ', 1)
    except ValueError:
        raise SystemExit('Unrecognized checksum line: ' + line[:120])
    # Reject dangerous and malformed paths. Preserve immutable v1.0.0 hashes.
    rel_path = Path(rel)
    if rel_path.is_absolute() or '..' in rel_path.parts:
        raise SystemExit('Unsafe checksum path: ' + rel)
    target = (ROOT / rel_path).resolve()
    if not target.is_file() or ROOT.resolve() not in target.parents:
        raise SystemExit('Checksum source is missing/outside repository: ' + rel)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest().lower() != expected_hash.strip().lower():
        raise SystemExit('SHA-256 mismatch: ' + rel)
    n += 1

# Check known public result CSV headers. A clean header does not guarantee
# privacy; human/legal review of all released material remains essential.
sensitive_columns = {'raw_text', 'clean_text', 'tweet_text', 'message_text', 'email', 'email_address'}
checked_headers = 0
for path in (ROOT / 'results').rglob('*.csv'):
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            cols = next(reader)
        except StopIteration:
            continue
    bad = sensitive_columns & {col.lower().strip() for col in cols}
    if bad:
        raise SystemExit(f'Potential restricted source columns in {path.relative_to(ROOT)}: {sorted(bad)}')
    checked_headers += 1

# The frozen release includes 4 intentionally cleaned Colab notebooks. Ensure
# they remain valid JSON; outputs are not treated as a guarantee of code safety.
notebooks = list((ROOT / 'notebooks').glob('*.ipynb'))
for path in notebooks:
    with path.open(encoding='utf-8') as f:
        nb = json.load(f)
    if 'cells' not in nb:
        raise SystemExit('Invalid notebook: ' + str(path))

print(f'PASS: {n} archived v1.0.0 file checksums match')
print(f'PASS: {len(EXPECTED)} expected research paths exist')
print(f'PASS: {checked_headers} result CSV headers checked for restricted raw-text fields')
print(f'PASS: {len(notebooks)} Colab notebooks are valid JSON')
print('Scope: provenance and saved predictions only, not model retraining.')
