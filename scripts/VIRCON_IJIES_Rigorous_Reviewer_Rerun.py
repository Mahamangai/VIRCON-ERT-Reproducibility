"""
VIRCON-Classify: IJIES reviewer-response rigorous rerun
======================================================

Purpose
-------
This script supplements the original VIRCON-Classify experiment code with the
specific analyses requested by the IJIES reviewers. It is intentionally
standalone with respect to the raw CrisisBench train/test files so that the
reviewer-facing outputs can be reproduced without relying on notebook state.

It adds:
1. Exact train/test and event-disjoint sample-accounting tables.
2. Exact one-to-one triage taxonomy export and hard validation.
3. Duplicate-safe group-agnostic and event-disjoint evaluations on the SAME
   strict identifiable-event population.
4. Per-class support, precision, recall and F1 for every model/protocol.
5. A modern frozen pretrained text-encoder baseline (MiniLM + balanced LR)
   evaluated under the same populations, folds, preprocessing and metrics.
6. End-to-end CPU latency and model-size measurement on the same machine.
7. Accuracy-latency-memory Pareto analysis.
8. Primary inferential analyses based on effect estimates + 95% CIs:
   - stratified paired message bootstrap for official fixed-split comparisons;
   - event-cluster bootstrap for event-disjoint comparisons;
   - Holm adjustment within explicitly defined hypothesis families.
   Seed-level Wilcoxon tests are deliberately NOT used as primary inference.
9. Event-construction sensitivity: verified alias-merged primary protocol vs a
   conservative sensitivity in which the two Typhoon Hagupit aliases are kept
   separate.
10. Coverage-qualified macro-F1 for labels represented in >=3 canonical events.
11. A stratified manual-audit template for the taxonomy-derived triage proxy.
12. Full reproducibility bundle: input hashes, split assignments, predictions,
    environment, dependency snapshot, config, manifests and checksums.

Important scientific boundary
-----------------------------
The event-disjoint analysis is a SAME-CORPUS event-shift evaluation. This code
must not be used to claim validated generalization to all unseen disasters.
An independent external corpus would be required for that stronger claim.

Run example (Google Colab / local Python)
-----------------------------------------
    pip install -U sentence-transformers torch scikit-learn pandas numpy scipy joblib psutil
    export VIRCON_TRAIN_PATH=crisis_consolidated_humanitarian_filtered_lang_en_train.tsv
    export VIRCON_TEST_PATH=crisis_consolidated_humanitarian_filtered_lang_en_test.tsv
    python VIRCON_IJIES_Rigorous_Reviewer_Rerun.py

Set VIRCON_RUN_MODERN_ENCODER=0 only for a quick sparse-only debug run. The
reviewer-facing manuscript run should keep it at 1.
"""
from __future__ import annotations

import gc
import hashlib
import html
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Keep deployment timing comparable and reduce hidden parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None
from sklearn.svm import LinearSVC

try:
    import psutil
except ImportError:  # optional
    psutil = None


# =============================================================================
# 0. Configuration and guard rails
# =============================================================================
@dataclass(frozen=True)
class ReviewConfig:
    train_path: str
    test_path: str
    text_col: str = "text"
    label_col: str = "class_label"
    event_col: str = "event"
    artifact_dir: str = "artifacts_ijies_reviewer_revision"
    random_state: int = 20260928
    cv_splits: int = 5
    n_bootstrap: int = 2000
    hash_features: int = 2**16
    tfidf_max_features: int = 50000
    min_df: int = 2
    max_df: float = 0.95
    c: float = 1.0
    max_iter_lr: int = 30000
    max_iter_svc: int = 40000
    coverage_min_events: int = 3
    manual_audit_per_source_label: int = 25
    run_modern_encoder: bool = True
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_batch_size: int = 128
    encoder_device: str = "cpu"
    deployment_single_n: int = 200
    deployment_batch_n: int = 1000
    deployment_batch_repeats: int = 5


CFG = ReviewConfig(
    train_path=os.environ.get(
        "VIRCON_TRAIN_PATH",
        "crisis_consolidated_humanitarian_filtered_lang_en_train.tsv",
    ),
    test_path=os.environ.get(
        "VIRCON_TEST_PATH",
        "crisis_consolidated_humanitarian_filtered_lang_en_test.tsv",
    ),
    text_col=os.environ.get("VIRCON_TEXT_COL", "text"),
    label_col=os.environ.get("VIRCON_LABEL_COL", "class_label"),
    event_col=os.environ.get("VIRCON_EVENT_COL", "event"),
    artifact_dir=os.environ.get(
        "VIRCON_REVIEW_ARTIFACT_DIR", "artifacts_ijies_reviewer_revision"
    ),
    random_state=int(os.environ.get("VIRCON_REVIEW_RANDOM_STATE", "20260928")),
    cv_splits=int(os.environ.get("VIRCON_REVIEW_CV_SPLITS", "5")),
    n_bootstrap=int(os.environ.get("VIRCON_REVIEW_N_BOOTSTRAP", "2000")),
    run_modern_encoder=os.environ.get("VIRCON_RUN_MODERN_ENCODER", "1") == "1",
    encoder_model=os.environ.get(
        "VIRCON_ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ),
    encoder_batch_size=int(os.environ.get("VIRCON_ENCODER_BATCH_SIZE", "128")),
    encoder_device=os.environ.get("VIRCON_ENCODER_DEVICE", "cpu"),
)

ROOT = Path(CFG.artifact_dir)
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for d in (ROOT, TABLES, REPORTS, MODELS):
    d.mkdir(parents=True, exist_ok=True)

random.seed(CFG.random_state)
np.random.seed(CFG.random_state)


# =============================================================================
# 1. Text normalization, I/O and hashing
# =============================================================================
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
USER_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]+)")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
NON_TOKEN_RE = re.compile(r"[^a-z0-9_\s']+")
SPACE_RE = re.compile(r"\s+")
TOKEN_PATTERN = r"(?u)\b[a-z][a-z0-9_]{1,}\b"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = html.unescape(str(value)).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = USER_RE.sub(" usertoken ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = NON_TOKEN_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Required dataset not found: {p.resolve()}\n"
            "Set VIRCON_TRAIN_PATH and VIRCON_TEST_PATH to the exact CrisisBench files."
        )
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported input format: {suffix}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def prepare_split(df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = [CFG.text_col, CFG.label_col, CFG.event_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{split_name} is missing required columns: {missing}")

    out = df.copy()
    raw_n = len(out)
    missing_label = out[CFG.label_col].isna() | (
        out[CFG.label_col].astype(str).str.strip() == ""
    )
    removed_missing_label = int(missing_label.sum())
    out = out.loc[~missing_label].copy()
    out["raw_text"] = out[CFG.text_col].fillna("").astype(str)
    out["clean_text"] = out["raw_text"].map(normalize_text)
    empty_text = out["clean_text"].eq("")
    removed_empty_text = int(empty_text.sum())
    out = out.loc[~empty_text].copy()
    out[CFG.label_col] = out[CFG.label_col].astype(str).str.strip()
    out["source_split"] = split_name
    out["source_row"] = np.arange(len(out), dtype=np.int64)
    out.reset_index(drop=True, inplace=True)
    audit = {
        "raw_rows": int(raw_n),
        "removed_missing_label": removed_missing_label,
        "removed_empty_normalized_text": removed_empty_text,
        "retained_rows": int(len(out)),
    }
    return out, audit


train_raw = read_table(CFG.train_path)
test_raw = read_table(CFG.test_path)
train_df, train_audit = prepare_split(train_raw, "train")
test_df, test_audit = prepare_split(test_raw, "test")
all_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
all_df["global_row_id"] = np.arange(len(all_df), dtype=np.int64)

LABELS = sorted(all_df[CFG.label_col].astype(str).unique().tolist())
if len(LABELS) != 11:
    warnings.warn(
        f"Expected 11 humanitarian labels but found {len(LABELS)}: {LABELS}",
        RuntimeWarning,
    )


# =============================================================================
# 2. Exact triage taxonomy and manual-audit template
# =============================================================================
REQUEST_SET = {"requests_or_needs"}
RESPONSE_SET = {"response_efforts", "donation_and_volunteering"}
TRIAGE_LABELS = ["other", "request", "response"]

missing_taxonomy_labels = sorted((REQUEST_SET | RESPONSE_SET) - set(LABELS))
if missing_taxonomy_labels:
    raise RuntimeError(
        "The configured triage mapping does not match the dataset. Missing source labels: "
        + ", ".join(missing_taxonomy_labels)
        + ". Inspect the exported source label list before changing the mapping."
    )


def triage_map(label: str) -> str:
    if label in REQUEST_SET:
        return "request"
    if label in RESPONSE_SET:
        return "response"
    return "other"


all_df["label_11"] = all_df[CFG.label_col].astype(str)
all_df["label_triage"] = all_df["label_11"].map(triage_map)
train_df = all_df.loc[all_df["source_split"].eq("train")].copy()
test_df = all_df.loc[all_df["source_split"].eq("test")].copy()

mapping_df = pd.DataFrame(
    {
        "source_label_exact": LABELS,
        "triage_label": [triage_map(x) for x in LABELS],
        "mapping_rule": [
            "explicit_request"
            if x in REQUEST_SET
            else "explicit_response"
            if x in RESPONSE_SET
            else "other_by_prespecified_rule"
            for x in LABELS
        ],
    }
)
mapping_df.to_csv(TABLES / "triage_taxonomy_exact_mapping.csv", index=False)
pd.DataFrame({"source_label_exact": LABELS}).to_csv(
    TABLES / "source_label_inventory.csv", index=False
)

# Reviewer-requested human-audit pathway. This file is a template only; it does
# not fabricate practitioner annotations.
audit_candidates = all_df.loc[all_df["label_triage"].eq("other")].copy()
audit_parts: List[pd.DataFrame] = []
for source_label, group in audit_candidates.groupby("label_11", sort=True):
    n = min(CFG.manual_audit_per_source_label, len(group))
    if n > 0:
        audit_parts.append(group.sample(n=n, random_state=CFG.random_state))
manual_audit = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
if not manual_audit.empty:
    cols = [
        "global_row_id",
        "source_split",
        "source_row",
        CFG.event_col,
        "label_11",
        "label_triage",
        "raw_text",
    ]
    manual_audit = manual_audit[cols].copy()
    manual_audit["audited_triage_label"] = ""
    manual_audit["implicit_request"] = ""
    manual_audit["implicit_response"] = ""
    manual_audit["auditor_notes"] = ""
manual_audit.to_csv(TABLES / "triage_manual_audit_template.csv", index=False)


# =============================================================================
# 3. Event canonicalization and exact inclusion/exclusion accounting
# =============================================================================
def normalize_event_key(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


GENERIC_EVENTS = {
    "",
    "nan",
    "none",
    "unknown",
    "missing_event",
    "disaster_events",
}
GENERIC_KEYS = {normalize_event_key(x) for x in GENERIC_EVENTS}

# Primary map: only explicitly documented/verified alias merges from the prior
# audit workflow. The two Hagupit aliases are merged in the primary protocol.
PRIMARY_ALIAS_MAP_RAW: Dict[str, str] = {
    "2012_sandy_hurricane-ontopic": "2012_hurricane_sandy",
    "2012_us_sandy-hurricane-a143145": "2012_hurricane_sandy",
    "2012_us_sandy-hurricane-a144267": "2012_hurricane_sandy",
    "2013_west_texas": "2013_west_texas_explosion",
    "2013_west-texas_explosion": "2013_west_texas_explosion",
    "2013_alberta_floods-ontopic": "2013_alberta_floods",
    "2013_alberta_floods": "2013_alberta_floods",
    "2013_queensland_floods-ontopic": "2013_queensland_floods",
    "2013_queensland_floods": "2013_queensland_floods",
    "2013_boston_bombings-ontopic": "2013_boston_bombings",
    "2013_boston_bombings": "2013_boston_bombings",
    "2011_joplin_tornado-a121571": "2011_joplin_tornado",
    "2011_joplin_tornado-a131709": "2011_joplin_tornado",
    "2015_vanuatu_cyclone": "2015_vanuatu_cyclone_pam",
    "2015_vanuatu_cyclone-pam": "2015_vanuatu_cyclone_pam",
    "2014_philippines_typhoon": "2014_typhoon_hagupit",
    "2014_philippines_typhoon-hagupit": "2014_typhoon_hagupit",
}
PRIMARY_ALIAS_MAP = {
    normalize_event_key(k): normalize_event_key(v)
    for k, v in PRIMARY_ALIAS_MAP_RAW.items()
}


def canonicalize_event(value: Any, alias_map: Mapping[str, str]) -> Tuple[str, str]:
    key = normalize_event_key(value)
    if key in GENERIC_KEYS:
        return "unresolved_generic_event_pool", "generic_excluded"
    canonical = alias_map.get(key, key)
    status = "alias_merged" if canonical != key else "unchanged"
    return canonical, status


def build_strict_population(
    frame: pd.DataFrame,
    alias_map: Mapping[str, str],
    tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    pairs = work[CFG.event_col].map(lambda x: canonicalize_event(x, alias_map))
    work["canonical_event"] = [p[0] for p in pairs]
    work["event_resolution"] = [p[1] for p in pairs]
    work["original_event"] = work[CFG.event_col].fillna("").astype(str)

    generic_mask = work["event_resolution"].eq("generic_excluded")
    identifiable = work.loc[~generic_mask].copy()

    text_event_counts = identifiable.groupby("clean_text")["canonical_event"].nunique()
    cross_event_texts = set(text_event_counts[text_event_counts > 1].index.astype(str))
    text_label_counts = identifiable.groupby("clean_text")["label_11"].nunique()
    conflicting_texts = set(text_label_counts[text_label_counts > 1].index.astype(str))

    identifiable["exclude_cross_event_duplicate_text"] = identifiable["clean_text"].isin(
        cross_event_texts
    )
    identifiable["exclude_conflicting_label_text"] = identifiable["clean_text"].isin(
        conflicting_texts
    )
    exclusion_union = identifiable[
        "exclude_cross_event_duplicate_text"
    ] | identifiable["exclude_conflicting_label_text"]
    strict = identifiable.loc[~exclusion_union].copy().reset_index(drop=True)

    both = int(
        (
            identifiable["exclude_cross_event_duplicate_text"]
            & identifiable["exclude_conflicting_label_text"]
        ).sum()
    )
    cross_only = int(
        (
            identifiable["exclude_cross_event_duplicate_text"]
            & ~identifiable["exclude_conflicting_label_text"]
        ).sum()
    )
    conflict_only = int(
        (
            ~identifiable["exclude_cross_event_duplicate_text"]
            & identifiable["exclude_conflicting_label_text"]
        ).sum()
    )

    flow = pd.DataFrame(
        [
            {
                "protocol": tag,
                "stage": "prepared_train_plus_test",
                "rows_retained": int(len(work)),
                "rows_removed_at_stage": 0,
            },
            {
                "protocol": tag,
                "stage": "exclude_generic_or_unresolved_event_pool",
                "rows_retained": int(len(identifiable)),
                "rows_removed_at_stage": int(generic_mask.sum()),
            },
            {
                "protocol": tag,
                "stage": "exclude_cross_event_duplicate_and_or_conflicting_texts",
                "rows_retained": int(len(strict)),
                "rows_removed_at_stage": int(exclusion_union.sum()),
            },
        ]
    )
    details = pd.DataFrame(
        [
            {"protocol": tag, "exclusion_category": "generic_event_pool", "rows": int(generic_mask.sum())},
            {"protocol": tag, "exclusion_category": "cross_event_duplicate_only", "rows": cross_only},
            {"protocol": tag, "exclusion_category": "conflicting_label_only", "rows": conflict_only},
            {"protocol": tag, "exclusion_category": "both_duplicate_and_conflict", "rows": both},
            {"protocol": tag, "exclusion_category": "final_strict_population", "rows": int(len(strict))},
        ]
    )

    excluded_rows = pd.concat(
        [
            work.loc[generic_mask].assign(exclusion_reason="generic_event_pool"),
            identifiable.loc[exclusion_union].assign(
                exclusion_reason=np.where(
                    identifiable.loc[exclusion_union, "exclude_cross_event_duplicate_text"]
                    & identifiable.loc[exclusion_union, "exclude_conflicting_label_text"],
                    "cross_event_duplicate_and_conflicting_label",
                    np.where(
                        identifiable.loc[
                            exclusion_union, "exclude_cross_event_duplicate_text"
                        ],
                        "cross_event_duplicate_text",
                        "conflicting_label_text",
                    ),
                )
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    return strict, flow, details, excluded_rows


strict_primary, flow_primary, exclusion_detail_primary, excluded_primary = build_strict_population(
    all_df, PRIMARY_ALIAS_MAP, "primary_alias_merged"
)
flow_primary.to_csv(TABLES / "event_disjoint_sample_flow.csv", index=False)
exclusion_detail_primary.to_csv(
    TABLES / "event_disjoint_exclusion_detail.csv", index=False
)
excluded_primary.to_csv(TABLES / "event_disjoint_excluded_rows.csv", index=False)

# Explicit overall sample accounting including raw/cleaning stages.
full_flow = pd.DataFrame(
    [
        {
            "stage": "raw_official_train",
            "rows": int(train_audit["raw_rows"]),
            "removed_from_previous": 0,
            "notes": "Official CrisisBench training partition before cleaning",
        },
        {
            "stage": "prepared_train",
            "rows": int(train_audit["retained_rows"]),
            "removed_from_previous": int(
                train_audit["removed_missing_label"]
                + train_audit["removed_empty_normalized_text"]
            ),
            "notes": "Removed missing labels or empty normalized text only",
        },
        {
            "stage": "raw_official_test",
            "rows": int(test_audit["raw_rows"]),
            "removed_from_previous": 0,
            "notes": "Official CrisisBench test partition before cleaning",
        },
        {
            "stage": "prepared_test",
            "rows": int(test_audit["retained_rows"]),
            "removed_from_previous": int(
                test_audit["removed_missing_label"]
                + test_audit["removed_empty_normalized_text"]
            ),
            "notes": "Removed missing labels or empty normalized text only",
        },
        {
            "stage": "prepared_combined",
            "rows": int(len(all_df)),
            "removed_from_previous": 0,
            "notes": "Prepared train + test; basis for same-corpus event-shift evaluation",
        },
        {
            "stage": "strict_identifiable_event_population",
            "rows": int(len(strict_primary)),
            "removed_from_previous": int(len(all_df) - len(strict_primary)),
            "notes": "Generic event pool and leakage-risk duplicate/conflicting texts excluded",
        },
    ]
)
full_flow.to_csv(TABLES / "complete_sample_accounting.csv", index=False)

# Alias map and event/class distributions used by the primary protocol.
alias_inventory = (
    strict_primary.groupby(["original_event", "canonical_event"], as_index=False)
    .size()
    .rename(columns={"size": "rows"})
)
alias_inventory.to_csv(TABLES / "canonical_event_map_and_counts.csv", index=False)

event_counts = (
    strict_primary.groupby("canonical_event", as_index=False)
    .agg(
        rows=("global_row_id", "size"),
        request_rows=("label_triage", lambda s: int(np.sum(np.asarray(s) == "request"))),
        response_rows=("label_triage", lambda s: int(np.sum(np.asarray(s) == "response"))),
    )
    .sort_values("rows", ascending=False)
)
event_counts.to_csv(TABLES / "canonical_event_counts.csv", index=False)

class_event_coverage = (
    strict_primary.groupby("label_11")["canonical_event"]
    .nunique()
    .reindex(LABELS, fill_value=0)
    .rename("canonical_events_with_label")
    .reset_index()
)
class_event_coverage.to_csv(TABLES / "class_event_coverage.csv", index=False)


# =============================================================================
# 4. Metrics, models and deterministic fold construction
# =============================================================================
def fixed_label_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    labels_list = list(labels)
    supported = [label for label in labels_list if np.any(yt == label)]
    out = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1_fixed_labels": float(
            f1_score(yt, yp, labels=labels_list, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(yt, yp, labels=labels_list, average="weighted", zero_division=0)
        ),
        "n_supported_labels": int(len(supported)),
    }
    out["balanced_accuracy_supported_labels"] = (
        float(balanced_accuracy_score(yt, yp)) if supported else float("nan")
    )
    out["macro_f1_supported_labels"] = (
        float(
            f1_score(yt, yp, labels=supported, average="macro", zero_division=0)
        )
        if supported
        else float("nan")
    )
    return out


def triage_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=TRIAGE_LABELS, zero_division=0
    )
    values = {
        label: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(TRIAGE_LABELS)
    }
    out: Dict[str, float] = {}
    for label in TRIAGE_LABELS:
        for metric, value in values[label].items():
            out[f"{label}_{metric}"] = value
    return out


def per_class_table(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    **meta: Any,
) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    rows = []
    for label in labels:
        d = report[str(label)]
        rows.append(
            {
                **meta,
                "label": label,
                "precision": float(d["precision"]),
                "recall": float(d["recall"]),
                "f1": float(d["f1-score"]),
                "support": int(d["support"]),
            }
        )
    return pd.DataFrame(rows)


def build_hash() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=CFG.hash_features,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        dtype=np.float64,
    )


def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=CFG.min_df,
        max_df=CFG.max_df,
        max_features=CFG.tfidf_max_features,
        lowercase=False,
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        norm="l2",
        dtype=np.float64,
    )


def make_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=CFG.c,
        solver="lbfgs",
        max_iter=CFG.max_iter_lr,
        tol=1e-4,
        class_weight="balanced",
        random_state=seed,
    )


def make_svc(seed: int) -> LinearSVC:
    return LinearSVC(
        C=CFG.c,
        class_weight="balanced",
        max_iter=CFG.max_iter_svc,
        random_state=seed,
    )


def make_group_agnostic_splits(frame: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = frame["label_11"].astype(str).to_numpy(dtype=object)
    duplicate_groups, _ = pd.factorize(frame["clean_text"].astype(str), sort=True)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=CFG.cv_splits, shuffle=True, random_state=CFG.random_state
        )
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in splitter.split(np.zeros(len(frame)), y, groups=duplicate_groups)
        ]
    else:
        warnings.warn(
            "StratifiedGroupKFold unavailable; falling back to GroupKFold grouped by clean_text.",
            RuntimeWarning,
        )
        splitter = GroupKFold(n_splits=CFG.cv_splits)
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in splitter.split(np.zeros(len(frame)), y, groups=duplicate_groups)
        ]
    # Hard duplicate-leakage check.
    fold_id = np.full(len(frame), -1, dtype=int)
    for k, (_, te) in enumerate(splits, start=1):
        fold_id[te] = k
    chk = pd.DataFrame({"text": frame["clean_text"].astype(str), "fold": fold_id})
    if (chk.groupby("text")["fold"].nunique() > 1).any():
        raise RuntimeError("Duplicate normalized text crossed group-agnostic folds.")
    return splits


def make_event_splits(frame: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    y = frame["label_11"].astype(str).to_numpy(dtype=object)
    groups = frame["canonical_event"].astype(str).to_numpy(dtype=object)
    n_groups = len(np.unique(groups))
    n_splits = min(CFG.cv_splits, n_groups)
    if n_splits < 2:
        raise RuntimeError("At least two canonical events are required.")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=CFG.random_state + 11
        )
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in splitter.split(np.zeros(len(frame)), y, groups=groups)
        ]
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in splitter.split(np.zeros(len(frame)), y, groups=groups)
        ]
    # Hard event-disjoint check.
    for tr, te in splits:
        if set(groups[tr]).intersection(set(groups[te])):
            raise RuntimeError("An event appears in both train and test in an event-disjoint fold.")
    return splits


MODEL_NAMES = ["hash_lr", "tfidf_lr", "tfidf_svc"]
if CFG.run_modern_encoder:
    MODEL_NAMES.append("minilm_lr")


# =============================================================================
# 5. Modern frozen encoder (reviewer-requested current text-encoder baseline)
# =============================================================================
encoder = None
all_embeddings: Optional[np.ndarray] = None
encoder_saved_dir = MODELS / "minilm_encoder"

if CFG.run_modern_encoder:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Modern encoder baseline requested but dependencies are missing. Run:\n"
            "pip install -U sentence-transformers torch\n"
            "Then rerun with VIRCON_RUN_MODERN_ENCODER=1."
        ) from exc

    torch.set_num_threads(1)
    encoder = SentenceTransformer(CFG.encoder_model, device=CFG.encoder_device)
    encoder.save(str(encoder_saved_dir))
    # Encode the prepared combined corpus once. The encoder is frozen and
    # pretrained externally; no CrisisBench label/test information is used in
    # representation fitting.
    all_embeddings = encoder.encode(
        all_df["clean_text"].astype(str).tolist(),
        batch_size=CFG.encoder_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(ROOT / "minilm_embeddings_all_prepared.npy", all_embeddings)


def embeddings_for(frame: pd.DataFrame) -> np.ndarray:
    if all_embeddings is None:
        raise RuntimeError("Modern embeddings requested but encoder baseline is disabled.")
    ids = frame["global_row_id"].to_numpy(dtype=np.int64)
    return all_embeddings[ids]


# =============================================================================
# 6. Fixed-split experiment under a priori settings (no test-informed tuning)
# =============================================================================
def fit_predict_sparse(
    model_name: str,
    train_text: Sequence[str],
    test_text: Sequence[str],
    y_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Any, Any]:
    if model_name == "hash_lr":
        vec = build_hash()
        Xtr = vec.transform(train_text).tocsr()
        Xte = vec.transform(test_text).tocsr()
        clf = make_lr(seed)
    elif model_name == "tfidf_lr":
        vec = build_tfidf()
        Xtr = vec.fit_transform(train_text).tocsr()
        Xte = vec.transform(test_text).tocsr()
        clf = make_lr(seed)
    elif model_name == "tfidf_svc":
        vec = build_tfidf()
        Xtr = vec.fit_transform(train_text).tocsr()
        Xte = vec.transform(test_text).tocsr()
        clf = make_svc(seed)
    else:
        raise KeyError(model_name)
    clf.fit(Xtr, y_train)
    pred = np.asarray(clf.predict(Xte), dtype=object)
    return pred, vec, clf


def fit_predict_embedding(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, LogisticRegression]:
    Xtr = embeddings_for(train_frame)
    Xte = embeddings_for(test_frame)
    clf = make_lr(seed)
    clf.fit(Xtr, y_train)
    return np.asarray(clf.predict(Xte), dtype=object), clf


fixed_rows: List[Dict[str, Any]] = []
fixed_per_class: List[pd.DataFrame] = []
fixed_predictions: Dict[Tuple[str, str], np.ndarray] = {}
fixed_fitted: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

for task, target_col, task_labels in (
    ("11class", "label_11", LABELS),
    ("direct_triage", "label_triage", TRIAGE_LABELS),
):
    ytr = train_df[target_col].astype(str).to_numpy(dtype=object)
    yte = test_df[target_col].astype(str).to_numpy(dtype=object)
    for model_name in MODEL_NAMES:
        if model_name == "minilm_lr":
            pred, clf = fit_predict_embedding(
                train_df, test_df, ytr, CFG.random_state
            )
            fitted = (None, clf)
        else:
            pred, vec, clf = fit_predict_sparse(
                model_name,
                train_df["clean_text"].astype(str).tolist(),
                test_df["clean_text"].astype(str).tolist(),
                ytr,
                CFG.random_state,
            )
            fitted = (vec, clf)
        metrics = fixed_label_metrics(yte, pred, task_labels)
        if task == "direct_triage":
            metrics.update(triage_metrics(yte, pred))
        fixed_rows.append(
            {
                "protocol": "official_fixed_split",
                "task": task,
                "model": model_name,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                **metrics,
            }
        )
        fixed_per_class.append(
            per_class_table(
                yte,
                pred,
                task_labels,
                protocol="official_fixed_split",
                task=task,
                model=model_name,
            )
        )
        fixed_predictions[(task, model_name)] = pred
        fixed_fitted[(task, model_name)] = fitted

fixed_summary = pd.DataFrame(fixed_rows)
fixed_summary.to_csv(TABLES / "fixed_split_summary_reviewer_rerun.csv", index=False)
pd.concat(fixed_per_class, ignore_index=True).to_csv(
    TABLES / "fixed_split_per_class_reviewer_rerun.csv", index=False
)

fixed_prediction_frame = test_df[
    ["global_row_id", "source_split", "source_row", CFG.event_col, "label_11", "label_triage"]
].copy()
for (task, model_name), pred in fixed_predictions.items():
    fixed_prediction_frame[f"pred_{task}__{model_name}"] = pred
fixed_prediction_frame.to_csv(TABLES / "fixed_split_predictions.csv", index=False)


# =============================================================================
# 7. Matched group-agnostic and event-disjoint CV on identical strict records
# =============================================================================
def run_cv_protocol(
    frame: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    protocol_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    texts = frame["clean_text"].astype(str).tolist()
    y11 = frame["label_11"].astype(str).to_numpy(dtype=object)
    yt = frame["label_triage"].astype(str).to_numpy(dtype=object)
    oof_11 = {m: np.full(len(frame), None, dtype=object) for m in MODEL_NAMES}
    oof_t = {m: np.full(len(frame), None, dtype=object) for m in MODEL_NAMES}
    fold_rows: List[Dict[str, Any]] = []
    fold_assign = np.full(len(frame), -1, dtype=int)

    for fold, (tr, te) in enumerate(splits, start=1):
        fold_assign[te] = fold
        train_text = [texts[int(i)] for i in tr]
        test_text = [texts[int(i)] for i in te]

        # Fit sparse feature blocks once per fold.
        hash_vec = build_hash()
        Xh_tr = hash_vec.transform(train_text).tocsr()
        Xh_te = hash_vec.transform(test_text).tocsr()
        tfidf_vec = build_tfidf()
        Xt_tr = tfidf_vec.fit_transform(train_text).tocsr()
        Xt_te = tfidf_vec.transform(test_text).tocsr()

        for task, y, labels, store in (
            ("11class", y11, LABELS, oof_11),
            ("direct_triage", yt, TRIAGE_LABELS, oof_t),
        ):
            for model_name in MODEL_NAMES:
                if model_name == "hash_lr":
                    Xtr, Xte, clf = Xh_tr, Xh_te, make_lr(CFG.random_state + fold)
                elif model_name == "tfidf_lr":
                    Xtr, Xte, clf = Xt_tr, Xt_te, make_lr(CFG.random_state + fold)
                elif model_name == "tfidf_svc":
                    Xtr, Xte, clf = Xt_tr, Xt_te, make_svc(CFG.random_state + fold)
                elif model_name == "minilm_lr":
                    Xemb = embeddings_for(frame)
                    Xtr, Xte, clf = Xemb[tr], Xemb[te], make_lr(CFG.random_state + fold)
                else:
                    raise KeyError(model_name)
                clf.fit(Xtr, y[tr])
                pred = np.asarray(clf.predict(Xte), dtype=object)
                store[model_name][te] = pred
                metrics = fixed_label_metrics(y[te], pred, labels)
                if task == "direct_triage":
                    metrics.update(triage_metrics(y[te], pred))
                held_events = sorted(frame.iloc[te]["canonical_event"].astype(str).unique().tolist())
                fold_rows.append(
                    {
                        "protocol": protocol_name,
                        "fold": fold,
                        "task": task,
                        "model": model_name,
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                        "n_train_events": int(frame.iloc[tr]["canonical_event"].nunique()),
                        "n_test_events": int(frame.iloc[te]["canonical_event"].nunique()),
                        "held_out_events": " | ".join(held_events),
                        **metrics,
                    }
                )

        del Xh_tr, Xh_te, Xt_tr, Xt_te
        gc.collect()

    if np.any(fold_assign < 1):
        raise RuntimeError(f"{protocol_name}: some records were not assigned a fold.")
    for model_name in MODEL_NAMES:
        if any(x is None for x in oof_11[model_name]):
            raise RuntimeError(f"{protocol_name}: missing 11-class OOF predictions for {model_name}")
        if any(x is None for x in oof_t[model_name]):
            raise RuntimeError(f"{protocol_name}: missing triage OOF predictions for {model_name}")

    pooled_rows: List[Dict[str, Any]] = []
    per_class_frames: List[pd.DataFrame] = []
    out_pred = frame[
        [
            "global_row_id",
            "source_split",
            "source_row",
            "original_event",
            "canonical_event",
            "label_11",
            "label_triage",
        ]
    ].copy()
    out_pred["fold"] = fold_assign

    for task, y, labels, store in (
        ("11class", y11, LABELS, oof_11),
        ("direct_triage", yt, TRIAGE_LABELS, oof_t),
    ):
        for model_name in MODEL_NAMES:
            pred = np.asarray(store[model_name], dtype=object)
            metrics = fixed_label_metrics(y, pred, labels)
            if task == "direct_triage":
                metrics.update(triage_metrics(y, pred))
            pooled_rows.append(
                {
                    "protocol": protocol_name,
                    "task": task,
                    "model": model_name,
                    "n_oof": int(len(y)),
                    "n_canonical_events": int(frame["canonical_event"].nunique()),
                    **metrics,
                }
            )
            per_class_frames.append(
                per_class_table(
                    y,
                    pred,
                    labels,
                    protocol=protocol_name,
                    task=task,
                    model=model_name,
                )
            )
            out_pred[f"pred_{task}__{model_name}"] = pred

    assignment = frame[
        ["global_row_id", "canonical_event", "label_11", "label_triage"]
    ].copy()
    assignment["fold"] = fold_assign
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(pooled_rows),
        pd.concat(per_class_frames, ignore_index=True),
        out_pred,
        assignment,
    )


group_splits = make_group_agnostic_splits(strict_primary)
event_splits = make_event_splits(strict_primary)

(
    group_fold,
    group_pooled,
    group_per_class,
    group_oof,
    group_assignment,
) = run_cv_protocol(strict_primary, group_splits, "group_agnostic_duplicate_safe")
(
    event_fold,
    event_pooled,
    event_per_class,
    event_oof,
    event_assignment,
) = run_cv_protocol(strict_primary, event_splits, "event_disjoint_same_corpus")

for name, df in {
    "group_agnostic_fold_results.csv": group_fold,
    "group_agnostic_pooled_summary.csv": group_pooled,
    "group_agnostic_per_class.csv": group_per_class,
    "group_agnostic_oof_predictions.csv": group_oof,
    "group_agnostic_fold_assignment.csv": group_assignment,
    "event_disjoint_fold_results.csv": event_fold,
    "event_disjoint_pooled_summary.csv": event_pooled,
    "event_disjoint_per_class.csv": event_per_class,
    "event_disjoint_oof_predictions.csv": event_oof,
    "event_disjoint_fold_assignment.csv": event_assignment,
}.items():
    df.to_csv(TABLES / name, index=False)

# Reviewer comment 4: explicit request-class support/precision/recall/F1 table.
request_table = event_per_class.loc[
    (event_per_class["task"] == "direct_triage")
    & (event_per_class["label"] == "request")
].copy()
request_table.to_csv(TABLES / "event_disjoint_request_class_results.csv", index=False)

# Per-fold class distributions.
fold_support_rows: List[Dict[str, Any]] = []
for protocol_name, frame, splits in (
    ("group_agnostic_duplicate_safe", strict_primary, group_splits),
    ("event_disjoint_same_corpus", strict_primary, event_splits),
):
    for fold, (tr, te) in enumerate(splits, start=1):
        for split_name, idx in (("train", tr), ("test", te)):
            for task, col, labels in (
                ("11class", "label_11", LABELS),
                ("direct_triage", "label_triage", TRIAGE_LABELS),
            ):
                vals = frame.iloc[idx][col].astype(str).to_numpy(dtype=object)
                for label in labels:
                    fold_support_rows.append(
                        {
                            "protocol": protocol_name,
                            "fold": fold,
                            "split": split_name,
                            "task": task,
                            "label": label,
                            "support": int(np.sum(vals == label)),
                        }
                    )
pd.DataFrame(fold_support_rows).to_csv(
    TABLES / "per_fold_class_support.csv", index=False
)


# =============================================================================
# 8. Event-construction sensitivity and coverage-qualified macro-F1
# =============================================================================
SENSITIVITY_ALIAS_MAP_RAW = dict(PRIMARY_ALIAS_MAP_RAW)
SENSITIVITY_ALIAS_MAP_RAW.pop("2014_philippines_typhoon", None)
SENSITIVITY_ALIAS_MAP_RAW.pop("2014_philippines_typhoon-hagupit", None)
SENSITIVITY_ALIAS_MAP = {
    normalize_event_key(k): normalize_event_key(v)
    for k, v in SENSITIVITY_ALIAS_MAP_RAW.items()
}
strict_sens, flow_sens, exclusion_sens, _ = build_strict_population(
    all_df, SENSITIVITY_ALIAS_MAP, "hagupit_aliases_kept_separate"
)
flow_sens.to_csv(TABLES / "sensitivity_sample_flow_hagupit_separate.csv", index=False)
exclusion_sens.to_csv(
    TABLES / "sensitivity_exclusion_detail_hagupit_separate.csv", index=False
)

# To keep sensitivity computationally focused, rerun the strongest sparse model
# and the modern encoder (when enabled) under the alternative event grouping.
sens_models = ["tfidf_svc"] + (["minilm_lr"] if CFG.run_modern_encoder else [])
_sens_original_model_names = list(MODEL_NAMES)
MODEL_NAMES[:] = sens_models
sens_event_splits = make_event_splits(strict_sens)
_, sens_pooled, sens_per_class, sens_oof, sens_assignment = run_cv_protocol(
    strict_sens, sens_event_splits, "event_disjoint_hagupit_separate_sensitivity"
)
MODEL_NAMES[:] = _sens_original_model_names
sens_pooled.to_csv(TABLES / "event_grouping_sensitivity_summary.csv", index=False)
sens_per_class.to_csv(TABLES / "event_grouping_sensitivity_per_class.csv", index=False)
sens_oof.to_csv(TABLES / "event_grouping_sensitivity_oof.csv", index=False)
sens_assignment.to_csv(TABLES / "event_grouping_sensitivity_assignment.csv", index=False)

coverage_counts = (
    strict_primary.groupby("label_11")["canonical_event"]
    .nunique()
    .reindex(LABELS, fill_value=0)
)
coverage_labels = [
    label for label in LABELS if int(coverage_counts.loc[label]) >= CFG.coverage_min_events
]
coverage_rows: List[Dict[str, Any]] = []
for protocol_name, oof in (
    ("group_agnostic_duplicate_safe", group_oof),
    ("event_disjoint_same_corpus", event_oof),
):
    y = oof["label_11"].astype(str).to_numpy(dtype=object)
    for model_name in _sens_original_model_names:
        pred = oof[f"pred_11class__{model_name}"].astype(str).to_numpy(dtype=object)
        coverage_rows.append(
            {
                "protocol": protocol_name,
                "model": model_name,
                "minimum_events_per_label": CFG.coverage_min_events,
                "included_label_count": len(coverage_labels),
                "included_labels": " | ".join(coverage_labels),
                "full_11class_macro_f1": float(
                    f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
                ),
                "coverage_qualified_macro_f1": float(
                    f1_score(
                        y,
                        pred,
                        labels=coverage_labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
pd.DataFrame(coverage_rows).to_csv(
    TABLES / "coverage_qualified_11class_metrics.csv", index=False
)


# =============================================================================
# 9. Primary inferential analysis: effect estimates + CIs, no seed Wilcoxon
# =============================================================================
def metric_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
    )


def stratified_paired_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    labels: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y_true == label) for label in labels]
    observed = metric_macro_f1(y_true, pred_a, labels) - metric_macro_f1(
        y_true, pred_b, labels
    )
    diffs = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        sampled_parts = []
        for idx in class_indices:
            if len(idx):
                sampled_parts.append(rng.choice(idx, size=len(idx), replace=True))
        sample = np.concatenate(sampled_parts)
        diffs[b] = metric_macro_f1(y_true[sample], pred_a[sample], labels) - metric_macro_f1(
            y_true[sample], pred_b[sample], labels
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    left = (np.sum(diffs <= 0.0) + 1) / (n_bootstrap + 1)
    right = (np.sum(diffs >= 0.0) + 1) / (n_bootstrap + 1)
    return {
        "effect_a_minus_b": float(observed),
        "ci_2_5": float(lo),
        "ci_97_5": float(hi),
        "bootstrap_p_two_sided": float(min(1.0, 2 * min(left, right))),
        "n_resamples": int(n_bootstrap),
        "resampling_unit": "message_within_true_class_strata",
    }


def event_cluster_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    unique_groups = sorted(np.unique(groups).tolist())
    matrices_a = []
    matrices_b = []
    for g in unique_groups:
        mask = groups == g
        matrices_a.append(confusion_matrix(y_true[mask], pred_a[mask], labels=list(labels)))
        matrices_b.append(confusion_matrix(y_true[mask], pred_b[mask], labels=list(labels)))
    matrices_a = np.asarray(matrices_a, dtype=float)
    matrices_b = np.asarray(matrices_b, dtype=float)

    def f1_from_cm(cm: np.ndarray) -> float:
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)
        return float(f1.mean())

    observed = f1_from_cm(matrices_a.sum(axis=0)) - f1_from_cm(matrices_b.sum(axis=0))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=float)
    n_groups = len(unique_groups)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        diffs[b] = f1_from_cm(matrices_a[sampled].sum(axis=0)) - f1_from_cm(
            matrices_b[sampled].sum(axis=0)
        )
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    left = (np.sum(diffs <= 0.0) + 1) / (n_bootstrap + 1)
    right = (np.sum(diffs >= 0.0) + 1) / (n_bootstrap + 1)
    return {
        "effect_a_minus_b": float(observed),
        "ci_2_5": float(lo),
        "ci_97_5": float(hi),
        "bootstrap_p_two_sided": float(min(1.0, 2 * min(left, right))),
        "n_resamples": int(n_bootstrap),
        "n_event_clusters": int(n_groups),
        "resampling_unit": "canonical_event_cluster",
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (len(p) - rank) * p[idx])
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


comparison_rows: List[Dict[str, Any]] = []
if CFG.run_modern_encoder:
    sparse_comparators = ["tfidf_svc", "tfidf_lr", "hash_lr"]
    # Fixed split families.
    for task, y_col, labels in (
        ("11class", "label_11", LABELS),
        ("direct_triage", "label_triage", TRIAGE_LABELS),
    ):
        y = test_df[y_col].astype(str).to_numpy(dtype=object)
        pred_a = fixed_predictions[(task, "minilm_lr")]
        for j, b_model in enumerate(sparse_comparators):
            result = stratified_paired_bootstrap_difference(
                y,
                pred_a,
                fixed_predictions[(task, b_model)],
                labels,
                CFG.n_bootstrap,
                CFG.random_state + 100 + j,
            )
            comparison_rows.append(
                {
                    "family": f"fixed_split_{task}_minilm_vs_sparse",
                    "protocol": "official_fixed_split",
                    "task": task,
                    "model_a": "minilm_lr",
                    "model_b": b_model,
                    "metric": "macro_f1_fixed_labels",
                    **result,
                }
            )

    # Event-disjoint families.
    event_groups = event_oof["canonical_event"].astype(str).to_numpy(dtype=object)
    for task, y_col, pred_prefix, labels in (
        ("11class", "label_11", "pred_11class__", LABELS),
        ("direct_triage", "label_triage", "pred_direct_triage__", TRIAGE_LABELS),
    ):
        y = event_oof[y_col].astype(str).to_numpy(dtype=object)
        pred_a = event_oof[f"{pred_prefix}minilm_lr"].astype(str).to_numpy(dtype=object)
        for j, b_model in enumerate(sparse_comparators):
            pred_b = event_oof[f"{pred_prefix}{b_model}"].astype(str).to_numpy(dtype=object)
            result = event_cluster_bootstrap_difference(
                y,
                pred_a,
                pred_b,
                event_groups,
                labels,
                CFG.n_bootstrap,
                CFG.random_state + 500 + j,
            )
            comparison_rows.append(
                {
                    "family": f"event_disjoint_{task}_minilm_vs_sparse",
                    "protocol": "event_disjoint_same_corpus",
                    "task": task,
                    "model_a": "minilm_lr",
                    "model_b": b_model,
                    "metric": "macro_f1_fixed_labels",
                    **result,
                }
            )

comparison_df = pd.DataFrame(comparison_rows)
if not comparison_df.empty:
    comparison_df["p_holm_within_family"] = np.nan
    for family, idx in comparison_df.groupby("family").groups.items():
        comparison_df.loc[idx, "p_holm_within_family"] = holm_adjust(
            comparison_df.loc[idx, "bootstrap_p_two_sided"].to_numpy()
        )
comparison_df.to_csv(TABLES / "primary_effect_estimates_and_cis.csv", index=False)

hypothesis_plan = pd.DataFrame(
    [
        {
            "family": "fixed_split_11class_minilm_vs_sparse",
            "pairing_unit": "same test message",
            "resampling": "stratified paired message bootstrap",
            "primary_quantity": "macro-F1 difference with 95% CI",
            "multiplicity": "Holm within this family",
        },
        {
            "family": "fixed_split_direct_triage_minilm_vs_sparse",
            "pairing_unit": "same test message",
            "resampling": "stratified paired message bootstrap",
            "primary_quantity": "macro-F1 difference with 95% CI",
            "multiplicity": "Holm within this family",
        },
        {
            "family": "event_disjoint_11class_minilm_vs_sparse",
            "pairing_unit": "same strict-corpus record",
            "resampling": "canonical-event cluster bootstrap",
            "primary_quantity": "macro-F1 difference with 95% CI",
            "multiplicity": "Holm within this family",
        },
        {
            "family": "event_disjoint_direct_triage_minilm_vs_sparse",
            "pairing_unit": "same strict-corpus record",
            "resampling": "canonical-event cluster bootstrap",
            "primary_quantity": "macro-F1 difference with 95% CI",
            "multiplicity": "Holm within this family",
        },
    ]
)
hypothesis_plan.to_csv(TABLES / "statistical_hypothesis_families.csv", index=False)


# =============================================================================
# 10. End-to-end CPU deployability and Pareto frontier
# =============================================================================
def directory_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024**2)


def benchmark_callable(fn, texts: Sequence[str]) -> Dict[str, float]:
    if not texts:
        return {"median_single_ms": float("nan"), "p95_single_ms": float("nan"), "batch_ms_per_message": float("nan")}
    # Warm-up.
    for text in list(texts[: min(10, len(texts))]):
        fn([text])
    single_times = []
    for text in list(texts[: min(CFG.deployment_single_n, len(texts))]):
        start = time.perf_counter()
        fn([text])
        single_times.append((time.perf_counter() - start) * 1000.0)
    batch_texts = list(texts[: min(CFG.deployment_batch_n, len(texts))])
    batch_times = []
    for _ in range(CFG.deployment_batch_repeats):
        start = time.perf_counter()
        fn(batch_texts)
        batch_times.append((time.perf_counter() - start) * 1000.0 / len(batch_texts))
    return {
        "median_single_ms": float(np.median(single_times)),
        "p95_single_ms": float(np.percentile(single_times, 95)),
        "batch_ms_per_message": float(np.median(batch_times)),
    }


# Persist fixed-split 11-class packages and benchmark raw-text end-to-end inference.
deploy_rows: List[Dict[str, Any]] = []
raw_test_texts = test_df["raw_text"].astype(str).tolist()
fixed_11 = fixed_summary.loc[fixed_summary["task"].eq("11class")].set_index("model")

for model_name in MODEL_NAMES:
    vec, clf = fixed_fitted[("11class", model_name)]
    package_dir = MODELS / f"deploy_{model_name}"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "minilm_lr":
        if encoder is None:
            continue
        joblib.dump(clf, package_dir / "classifier.joblib", compress=3)
        # Encoder weights are counted from the shared saved encoder directory.
        model_size = directory_size_mb(encoder_saved_dir) + (
            (package_dir / "classifier.joblib").stat().st_size / (1024**2)
        )

        def predict_fn(batch: Sequence[str], _clf=clf):
            cleaned = [normalize_text(x) for x in batch]
            emb = encoder.encode(
                cleaned,
                batch_size=CFG.encoder_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return _clf.predict(emb)

    else:
        joblib.dump({"vectorizer": vec, "classifier": clf}, package_dir / "pipeline.joblib", compress=3)
        model_size = (package_dir / "pipeline.joblib").stat().st_size / (1024**2)

        def predict_fn(batch: Sequence[str], _vec=vec, _clf=clf):
            cleaned = [normalize_text(x) for x in batch]
            X = _vec.transform(cleaned)
            return _clf.predict(X)

    timing = benchmark_callable(predict_fn, raw_test_texts)
    row = fixed_11.loc[model_name]
    deploy_rows.append(
        {
            "model": model_name,
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1_fixed_labels"]),
            "pipeline_size_mb": float(model_size),
            **timing,
            "device": CFG.encoder_device if model_name == "minilm_lr" else "cpu",
            "thread_policy": "single-thread numerical libraries requested",
        }
    )

deploy_df = pd.DataFrame(deploy_rows)

def is_dominated(i: int) -> bool:
    a = deploy_df.iloc[i]
    for j, b in deploy_df.iterrows():
        if j == i:
            continue
        no_worse = (
            b["macro_f1"] >= a["macro_f1"]
            and b["median_single_ms"] <= a["median_single_ms"]
            and b["pipeline_size_mb"] <= a["pipeline_size_mb"]
        )
        strictly_better = (
            b["macro_f1"] > a["macro_f1"]
            or b["median_single_ms"] < a["median_single_ms"]
            or b["pipeline_size_mb"] < a["pipeline_size_mb"]
        )
        if no_worse and strictly_better:
            return True
    return False

if not deploy_df.empty:
    deploy_df["pareto_dominated_3d"] = [is_dominated(i) for i in range(len(deploy_df))]
deploy_df.to_csv(TABLES / "accuracy_latency_memory_pareto.csv", index=False)


# =============================================================================
# 11. A-priori model-selection protocol and claim-scope report
# =============================================================================
selection_protocol = {
    "hyperparameter_selection": "No test-set hyperparameter tuning in this reviewer rerun.",
    "sparse_hyperparameters": {
        "ngram_range": [1, 2],
        "tfidf_min_df": CFG.min_df,
        "tfidf_max_df": CFG.max_df,
        "tfidf_max_features": CFG.tfidf_max_features,
        "hash_features": CFG.hash_features,
        "classifier_C": CFG.c,
        "class_weight": "balanced",
    },
    "modern_encoder": {
        "model_id": CFG.encoder_model,
        "frozen": True,
        "classifier": "balanced logistic regression",
        "classifier_C": CFG.c,
        "corpus_specific_fine_tuning": False,
        "representation_fit_on_test_labels": False,
    },
    "split_policy": {
        "official_fixed_split": "unchanged CrisisBench train/test partitions",
        "group_agnostic": "same strict records; normalized-text identity grouped to prevent duplicate leakage",
        "event_disjoint": "same strict records; canonical event wholly absent from training fold",
        "event_split_random_state": CFG.random_state + 11,
        "performance_used_to_choose_split": False,
    },
    "inference_policy": {
        "seed_level_Wilcoxon_primary": False,
        "fixed_split_primary": "stratified paired message bootstrap effect + 95% CI",
        "event_disjoint_primary": "canonical-event cluster bootstrap effect + 95% CI",
        "multiplicity": "Holm within explicitly declared hypothesis family",
    },
}
(REPORTS / "model_selection_and_inference_protocol.json").write_text(
    json.dumps(selection_protocol, indent=2), encoding="utf-8"
)

claim_scope = {
    "supported_claim": "computational deployability and offline screening performance under official split plus same-corpus event shift",
    "unsupported_claim": "validated generalization to arbitrary unseen disasters or demonstrated end-to-end emergency-management utility",
    "reason": "No independent external disaster corpus and no human-in-the-loop operational outcome study are included in this rerun.",
    "triage_proxy": "taxonomy-derived; practitioner/manual audit template exported but must be completed before claiming practitioner-validated triage",
}
(REPORTS / "claim_scope_and_limitations.json").write_text(
    json.dumps(claim_scope, indent=2), encoding="utf-8"
)


# =============================================================================
# 12. Reproducibility metadata, checksums, manifest and final ZIP
# =============================================================================
def environment_report() -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": __import__("sklearn").__version__,
        "logical_cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    if CFG.run_modern_encoder:
        import torch
        import sentence_transformers
        d.update(
            {
                "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "encoder_device": CFG.encoder_device,
            }
        )
    if psutil is not None:
        d.update(
            {
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            }
        )
    return d


(REPORTS / "config.json").write_text(json.dumps(asdict(CFG), indent=2), encoding="utf-8")
(REPORTS / "environment.json").write_text(
    json.dumps(environment_report(), indent=2), encoding="utf-8"
)
input_hashes = {
    "train_path": str(Path(CFG.train_path).resolve()),
    "train_sha256": sha256_file(Path(CFG.train_path)),
    "test_path": str(Path(CFG.test_path).resolve()),
    "test_sha256": sha256_file(Path(CFG.test_path)),
}
(REPORTS / "input_file_hashes.json").write_text(
    json.dumps(input_hashes, indent=2), encoding="utf-8"
)

try:
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.STDOUT
    )
except Exception as exc:  # pragma: no cover
    freeze = f"pip freeze unavailable: {exc}"
(REPORTS / "pip_freeze.txt").write_text(freeze, encoding="utf-8")

reviewer_coverage = pd.DataFrame(
    [
        ["R1.1", "Deployment claims / latency / size", "accuracy_latency_memory_pareto.csv", "Separates predictive quality, latency and package size on the same CPU."],
        ["R1.2", "Event-disjoint sample accounting", "complete_sample_accounting.csv + event_disjoint_exclusion_detail.csv", "Explicit row flow and exclusion reasons."],
        ["R1.3", "Exact triage taxonomy", "triage_taxonomy_exact_mapping.csv", "One source label -> one triage label with hard validation."],
        ["R1.4", "Request-class event-disjoint metrics", "event_disjoint_request_class_results.csv", "Explicit support/precision/recall/F1 for request class."],
        ["R1.5", "Model-selection protocol", "model_selection_and_inference_protocol.json", "A-priori settings; no test-informed tuning."],
        ["R1.6 / R2.4", "Modern text encoder and Pareto comparison", "fixed/event CV summaries + accuracy_latency_memory_pareto.csv", "Frozen MiniLM + balanced LR under matched protocols/hardware."],
        ["R1.7", "Statistical rigor", "primary_effect_estimates_and_cis.csv + statistical_hypothesis_families.csv", "Effect + CI primary; event-cluster bootstrap; Holm within family; no primary seed Wilcoxon."],
        ["R1.8", "Public reproducibility artifacts", "entire artifact bundle", "Hashes, config, environment, folds, predictions, per-class outputs, dependency snapshot."],
        ["R2.2", "Generalization scope", "claim_scope_and_limitations.json", "Restricts claim to same-corpus event shift unless external corpus is added."],
        ["R2.3", "Triage proxy validation", "triage_manual_audit_template.csv", "Creates blinded human-audit template; must be completed by authors/practitioners for validation claim."],
        ["R2.5", "Event-construction sensitivity", "event_grouping_sensitivity_summary.csv", "Primary verified alias merge vs Hagupit-kept-separate sensitivity."],
        ["R2.6", "Operational claim scope", "claim_scope_and_limitations.json", "Uses computational deployability/offline screening wording unless human workflow study is added."],
        ["R2.7", "Event shift as primary limitation", "matched group/event summaries + effect tables", "Same strict corpus evaluated under duplicate-safe and event-disjoint protocols."],
    ],
    columns=["reviewer_comment", "issue", "output", "how_addressed"],
)
reviewer_coverage.to_csv(TABLES / "reviewer_comment_code_coverage_matrix.csv", index=False)

# File-level artifact checksums.
manifest_rows = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file():
        manifest_rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(ROOT / "artifact_manifest_sha256.csv", index=False)

readme = f"""# VIRCON-Classify IJIES Reviewer-Revision Artifact Bundle

This directory was generated by `VIRCON_IJIES_Rigorous_Reviewer_Rerun.py`.

## Primary reviewer-facing outputs
- `tables/complete_sample_accounting.csv`
- `tables/triage_taxonomy_exact_mapping.csv`
- `tables/fixed_split_summary_reviewer_rerun.csv`
- `tables/group_agnostic_pooled_summary.csv`
- `tables/event_disjoint_pooled_summary.csv`
- `tables/event_disjoint_request_class_results.csv`
- `tables/per_fold_class_support.csv`
- `tables/event_grouping_sensitivity_summary.csv`
- `tables/coverage_qualified_11class_metrics.csv`
- `tables/primary_effect_estimates_and_cis.csv`
- `tables/statistical_hypothesis_families.csv`
- `tables/accuracy_latency_memory_pareto.csv`
- `tables/reviewer_comment_code_coverage_matrix.csv`
- `reports/model_selection_and_inference_protocol.json`
- `reports/claim_scope_and_limitations.json`
- `reports/input_file_hashes.json`
- `reports/environment.json`
- `reports/pip_freeze.txt`

## Manual triage audit
`tables/triage_manual_audit_template.csv` is intentionally NOT pre-annotated. Complete
`audited_triage_label`, `implicit_request`, `implicit_response`, and `auditor_notes`
with an independent human/practitioner audit before claiming validated operational triage.

## Interpretation boundary
The event-disjoint experiment is a same-corpus event-shift evaluation. It is not an
independent external-validation study and should not be described as proof of universal
unseen-disaster generalization.

## Modern baseline
Modern baseline enabled: {CFG.run_modern_encoder}
Encoder: {CFG.encoder_model}
Encoder policy: frozen pretrained representation + class-balanced logistic regression.
The same normalized text, strict record population, fold assignments and metrics are used
for matched sparse/encoder comparisons.
"""
(ROOT / "README_REVIEWER_REVISION.md").write_text(readme, encoding="utf-8")

completion = {
    "status": "completed",
    "prepared_train_rows": int(len(train_df)),
    "prepared_test_rows": int(len(test_df)),
    "strict_event_rows": int(len(strict_primary)),
    "strict_canonical_events": int(strict_primary["canonical_event"].nunique()),
    "models": _sens_original_model_names,
    "modern_encoder_enabled": bool(CFG.run_modern_encoder),
    "primary_inference_uses_seed_wilcoxon": False,
    "manual_triage_audit_completed": False,
    "external_validation_included": False,
}
(REPORTS / "completion_summary.json").write_text(
    json.dumps(completion, indent=2), encoding="utf-8"
)

# Refresh manifest after README/completion were written.
manifest_rows = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.name != "artifact_manifest_sha256.csv":
        manifest_rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
pd.DataFrame(manifest_rows).to_csv(ROOT / "artifact_manifest_sha256.csv", index=False)

zip_path = Path("VIRCON_IJIES_Reviewer_Revision_Artifacts.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for path in ROOT.rglob("*"):
        if path.is_file():
            zf.write(path, arcname=str(path.relative_to(ROOT.parent)))

print("\nIJIES REVIEWER-REVISION RERUN COMPLETED")
print("Artifact directory:", ROOT.resolve())
print("Final artifact ZIP:", zip_path.resolve())
print("Prepared train/test:", len(train_df), len(test_df))
print("Strict event population:", len(strict_primary))
print("Canonical events:", strict_primary["canonical_event"].nunique())
print("Modern encoder enabled:", CFG.run_modern_encoder)
print("Primary seed-level Wilcoxon inference used: NO")
print("Manual triage audit still requires human completion: YES")
print("External independent validation included: NO")
