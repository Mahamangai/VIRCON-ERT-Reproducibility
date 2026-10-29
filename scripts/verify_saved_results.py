#!/usr/bin/env python3
"""Verify headline VIRCON-ERT results from saved predictions (NO model retraining)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
BASE=Path(__file__).resolve().parents[1]
R=BASE/'results'

def csv(p):
    path = R / p
    if not path.exists() and (R / (p + '.gz')).exists():
        path = R / (p + '.gz')
    return pd.read_csv(path)
def check(label,actual,expected,tol=0.00011):
    assert abs(actual-expected) < tol, f'{label}: observed {actual:.6f}, expected {expected:.6f}'
    print(f'PASS  {label}: {actual:.6f}')

p=csv('vircon_ert_confirmation/vircon_ert_fixed_predictions.csv')
assert len(p)==17356
check('CrisisBench 11-class accuracy',accuracy_score(p.label_11,p.pred_11),0.825305)
check('CrisisBench 11-class macro-F1',f1_score(p.label_11,p.pred_11,average='macro',zero_division=0),0.685680)
check('CrisisBench triage macro-F1',f1_score(p.label_triage,p.pred_triage,average='macro',zero_division=0),0.859273)

b=csv('distilbert_and_external/distilbert_event_disjoint_oof_predictions.csv')
assert len(b)==67243
check('DistilBERT event macro-F1',f1_score(b.label_11,b.pred_11class__distilbert_ft,average='macro',zero_division=0),0.459323)

e=csv('architecture_ablation/erht_event_disjoint_predictions.csv')
e=e[e.model=='erht_no_hierarchy']
assert len(e)==67243 and e.canonical_event.nunique()==50
check('ERHT no-hierarchy event macro-F1',f1_score(e.label_11,e.pred_11,average='macro',zero_division=0),0.474278)

fm=pd.read_csv(BASE/'data_provenance/event_disjoint_fold_assignment.csv')
assert len(fm)==67243 and fm.canonical_event.nunique()==50 and fm.fold.nunique()==5
assert fm.global_row_id.is_unique
assert fm.groupby('canonical_event').fold.nunique().max()==1
print('PASS  50 events; 5 distinct event folds; no duplicate record IDs')

au=csv('manual_audit/triage_manual_audit_FINAL_public_safe.csv')
assert len(au)==200
check('Human audit raw agreement',float((au.rater1_triage==au.rater2_triage).mean()),0.965,1e-7)
check('Human audit Cohen kappa',cohen_kappa_score(au.rater1_triage,au.rater2_triage),0.8127,0.0002)

ex=pd.read_csv(BASE/'data_provenance/event_disjoint_exclusion_detail.csv')
assert ex.loc[ex.exclusion_category=='generic_event_pool','rows'].iloc[0]==11269
assert ex.loc[ex.exclusion_category=='final_strict_population','rows'].iloc[0]==67243
print('PASS  exclusion summary consistent')

n=csv('vircon_ert_confirmation/text_free_vircon_ert_humaid_predictions.csv')
assert len(n)==15160
check('HumAID triage macro-F1',f1_score(n.label_triage,n.pred_triage__vircon_ert,average='macro',zero_division=0),0.492058)

stat=csv('architecture_ablation/erht_ALL_variants_vs_distilbert_bootstrap.csv')
row=stat[(stat.task=='11class') & (stat.candidate=='erht_no_hierarchy')].iloc[0]
check('Event F1 effect',float(row.difference_candidate_minus_distilbert),0.014955,0.0001)
assert float(row.ci95_low)<0<float(row.ci95_high)
print('PASS  event-cluster CI includes zero; no significant gain claim')
print('ALL EXPORTED PREDICTION-CHECKS PASSED. Training is not independently replayed.')
