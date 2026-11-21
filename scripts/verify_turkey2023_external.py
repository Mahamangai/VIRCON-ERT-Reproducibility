#!/usr/bin/env python3
"""Verify archived Turkey-Syria 2023 external-evaluation tables (no model training).
Run from repository root: python scripts/verify_turkey2023_external.py
Requires: numpy, pandas, scikit-learn (see requirements-ci.txt).

This checks artifact hashes, confusion matrix metrics, request counts, per-class
metrics, population sizes and paired contrast point estimates. Bootstrap CIs
are archived (not recomputed) because the frozen ground-truth CSV is withheld.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

BASE = Path(__file__).resolve().parents[1] / 'results' / 'external_turkey_syria_2023'
BENCH = BASE / 'benchmark'
ANALYSIS = BASE / 'analysis'
MODELS = ('VIRCON_full', 'Crisis_CT_M1', 'TFIDF_LinearSVC')
POPS = {'all_600': 600, 'relevant_504': 504}
LABELS = (
    'affected_individual', 'caution_and_advice', 'displaced_and_evacuations',
    'donation_and_volunteering', 'infrastructure_and_utilities_damage',
    'injured_or_dead_people', 'missing_and_found_people', 'not_humanitarian',
    'requests_or_needs', 'response_efforts', 'sympathy_and_support'
)
TRIAGE = ('other', 'request', 'response')

def check_close(x, y, name, tol=1e-9):
    if not (np.isfinite(float(x)) and np.isfinite(float(y)) and abs(float(x)-float(y)) <= tol):
        raise AssertionError(f'{name}: {x} != {y}')

def verify_hashes():
    inventory=BASE/'NEW_FILES_SHA256SUMS.txt'
    count=0
    for raw in inventory.read_text().splitlines():
        if not raw.strip():
            continue
        sha, rel=raw.split('  ', 1)
        path=BASE/rel
        assert path.is_file(), f'Missing {path}'
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==sha, f'Checksum mismatch: {rel}'
        count+=1
    return count

def main():
    hashed=verify_hashes()
    report=pd.read_csv(BENCH/'External_Model_Comparison_3_Models.csv')
    audit=pd.read_csv(ANALYSIS/'Request_Alert_Audit.csv')
    boot=pd.read_csv(ANALYSIS/'Paired_Bootstrap_VIRCON_vs_Both_2000.csv')
    assert len(report)==6 and len(audit)==6 and len(boot)==4
    assert set(report.model)==set(MODELS)
    manifest=json.loads((BENCH/'External_Crisis_Benchmark_Manifest.json').read_text())
    assert manifest['external_labels_used_to_train'] is False
    assert manifest['seed']==20266008
    saved_preds={}
    for model in MODELS:
        pp=pd.read_csv(BENCH/f'{model}_Predictions_No_Text.csv',dtype={'message_id':str})
        assert len(pp)==600 and pp.message_id.is_unique
        assert {'message_id','pred_11','pred_triage'}.issubset(pp.columns)
        saved_preds[model]=set(pp.message_id)
    assert saved_preds[MODELS[0]]==saved_preds[MODELS[1]]==saved_preds[MODELS[2]]
    for model in MODELS:
        for pop,n in POPS.items():
            row=report.loc[(report.model==model)&(report.population==pop)]
            assert len(row)==1,(model,pop)
            row=row.iloc[0]
            assert int(row.n_humanitarian)==n
            cm=pd.read_csv(BENCH/f'{model}_{pop}_Confusion11.csv',index_col=0)
            assert cm.index.tolist()==list(LABELS)
            assert cm.columns.tolist()==list(LABELS)
            assert cm.to_numpy().sum()==n
            acc=float(np.trace(cm.to_numpy())/n)
            check_close(acc,row.accuracy_11, f'{model} {pop} accuracy')
            arr=cm.to_numpy(dtype=float)
            tp=np.diag(arr)
            precision=np.divide(tp,arr.sum(axis=0),out=np.zeros(11),where=arr.sum(axis=0)!=0)
            recall=np.divide(tp,arr.sum(axis=1),out=np.zeros(11),where=arr.sum(axis=1)!=0)
            f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(11),where=(precision+recall)!=0)
            check_close(f1.mean(),row.macro_f1_11,f'{model} {pop} macro-f1')
            per=pd.read_csv(BENCH/f'{model}_{pop}_PerClass11.csv')
            assert per['label'].tolist()==list(LABELS)
            for key,values in [('precision',precision),('recall',recall),('f1',f1),('support',arr.sum(axis=1))]:
                assert np.allclose(per[key].to_numpy(dtype=float),values,atol=1e-9,rtol=0), f'{model} {pop} {key}'
            tri=pd.read_csv(BENCH/f'{model}_{pop}_Confusion3.csv',index_col=0)
            assert tri.index.tolist()==list(TRIAGE)
            assert tri.columns.tolist()==list(TRIAGE)
            assert tri.to_numpy().sum()==int(row.n_actionability)
            yy=tri.to_numpy(dtype=float)
            t=np.diag(yy)
            pp=np.divide(t,yy.sum(axis=0),out=np.zeros(3),where=yy.sum(axis=0)!=0)
            rr=np.divide(t,yy.sum(axis=1),out=np.zeros(3),where=yy.sum(axis=1)!=0)
            ff=np.divide(2*pp*rr,pp+rr,out=np.zeros(3),where=pp+rr!=0)
            check_close(ff.mean(),row.macro_f1_actionability,f'{model} {pop} triage-f1')
            req=TRIAGE.index('request')
            tp_req=int(yy[req,req]); fp_req=int(yy[:,req].sum()-tp_req);fn_req=int(yy[req,:].sum()-tp_req)
            assert (tp_req,fp_req,fn_req)==(int(row.request_TP),int(row.request_FP),int(row.request_FN))
            aud=audit.loc[(audit.model==model)&(audit.population==pop)]
            assert len(aud)==1
            aud=aud.iloc[0]
            assert int(aud.request_TP)==tp_req and int(aud.request_FP)==fp_req and int(aud.request_FN)==fn_req
            assert int(aud.excluded_uncertain_actionability)==5
    for _,r in boot.iterrows():
        pop=r['population']
        assert pop in POPS
        for target,name in [('Crisis_CT_M1','Crisis_CT_M1'),('TFIDF_LinearSVC','TFIDF_LinearSVC')]:
            if str(r['comparison'])=='VIRCON_full minus '+target:
                lhs=float(report.query('model == "VIRCON_full" and population == @pop').macro_f1_11.iloc[0])
                rhs=float(report.query('model == @target and population == @pop').macro_f1_11.iloc[0])
                check_close(lhs-rhs,r.macro_f1_difference,f'bootstrap point {pop} {target}')
                assert int(r.resamples)==2000
                assert float(r.CI95_low)<float(r.CI95_high)
                break
        else: raise AssertionError(f'Unexpected contrast {r["comparison"]}')
    ann=json.loads((ANALYSIS/'Annotation_Audit_Notes.json').read_text())
    assert ann['uncertain_actionability_count']==5 and ann['annotation_changes_made_by_this_cell']==0
    print(f'PASS: {hashed} output file SHA-256 hashes match.')
    print('PASS: all 6 model/population comparisons and 4 paired contrasts match saved summaries.')
    print('Limitations: this does not repeat GPU training, reproduce bootstrap draws, or check original labels.')

if __name__=='__main__':
    main()
