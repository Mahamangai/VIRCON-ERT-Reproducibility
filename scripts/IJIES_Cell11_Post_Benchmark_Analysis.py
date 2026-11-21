# CELL 11 — Paste below the completed external benchmark cells in the SAME Kaggle notebook.
# No neural-network training, no model downloads, no new-event label changes.
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

OUT11 = Path('/kaggle/working/IJIES_Post_Benchmark_Analysis')
OUT11.mkdir(parents=True, exist_ok=True)
BENCH11 = Path('/kaggle/working/IJIES_Crisis_External_Benchmark')
LABELS11 = [
    'affected_individual', 'caution_and_advice', 'displaced_and_evacuations',
    'donation_and_volunteering', 'infrastructure_and_utilities_damage',
    'injured_or_dead_people', 'missing_and_found_people', 'not_humanitarian',
    'requests_or_needs', 'response_efforts', 'sympathy_and_support'
]
TRIAGE11 = ['other', 'request', 'response']
FROZEN_SHA11 = 'c58378ba9b5b7ae247fb2b44eda79459ac76b7712f3df876ee8bb8198e9ff1b5'

def _candidate_files11(name):
    found=[]
    for root in [Path('/kaggle/working'), Path('/kaggle/input')]:
        if root.exists():
            found.extend(p for p in root.rglob(name) if p.is_file())
    return sorted(set(found), key=lambda p: (len(str(p)),str(p)))

def _hash11(path):
    h=hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

# Reuse the already-loaded frozen dataset; otherwise read its frozen CSV only.
if 'frozen' in globals() and isinstance(frozen, pd.DataFrame):
    audit11=frozen.copy()
    print('Using frozen ground truth already held in this Kaggle notebook.')
else:
    candidates11=_candidate_files11('IJIES_Independent_Event_FROZEN_600.csv')
    if not candidates11:
        raise FileNotFoundError('Frozen ground truth missing. Run the ORIGINAL Cell 1 only, or attach IJIES_Independent_Event_FROZEN_600.csv via Add Input. Do not rerun model training.')
    frozen_file11=next((p for p in candidates11 if _hash11(p)==FROZEN_SHA11), None)
    if frozen_file11 is None:
        raise ValueError('Frozen CSV does not match the prior benchmark SHA256; stop before statistical comparisons.')
    audit11=pd.read_csv(frozen_file11, dtype=str).fillna('')
    print('Loaded and SHA256-verified:',frozen_file11)

# If the runtime still has the actual frozen file path, verify it again.
if 'frozen_path' in globals() and Path(frozen_path).is_file():
    assert _hash11(Path(frozen_path))==FROZEN_SHA11, 'Frozen file checksum does not match the published comparison.'

mandatory11={'message_id','humanitarian_label_11','actionability_label','disaster_relevance'}
assert mandatory11.issubset(audit11.columns), f'Missing frozen columns: {mandatory11-set(audit11.columns)}'
for col in mandatory11:
    audit11[col]=audit11[col].astype(str).str.strip()
assert len(audit11)==600 and audit11.message_id.is_unique
assert audit11.humanitarian_label_11.isin(LABELS11).all()
assert audit11.actionability_label.isin(TRIAGE11+['uncertain']).all()
assert audit11.disaster_relevance.isin(['relevant','not_relevant','uncertain']).all()
assert int(audit11.disaster_relevance.eq('relevant').sum())==504

archives11=_candidate_files11('IJIES_Crisis_External_Benchmark_Results_No_Text.zip')
models11={
    'VIRCON_full':['VIRCON_full_External_Predictions_No_Text.csv','VIRCON_full_Predictions_No_Text.csv'],
    'Crisis_CT_M1':['Crisis_CT_M1_External_Predictions_No_Text.csv','Crisis_CT_M1_Predictions_No_Text.csv'],
    'TFIDF_LinearSVC':['TFIDF_LinearSVC_External_Predictions_No_Text.csv','TFIDF_LinearSVC_Predictions_No_Text.csv']
}
preds11={}
for model, names in models11.items():
    frame=None
    for name in names:
        direct=BENCH11/name
        hits=[direct] if direct.is_file() else _candidate_files11(name)
        if hits:
            frame=pd.read_csv(hits[0], dtype=str)
            break
    if frame is None:
        for zpath in archives11:
            with ZipFile(zpath) as z:
                found=next((n for n in z.namelist() if Path(n).name in names),None)
                if found:
                    frame=pd.read_csv(BytesIO(z.read(found)), dtype=str)
                    break
    if frame is None:
        raise FileNotFoundError(f'Missing {model} predictions. Keep prior Kaggle results, or Add Input: IJIES_Crisis_External_Benchmark_Results_No_Text.zip')
    needed={'message_id','pred_11','pred_triage'}
    assert needed.issubset(frame.columns) and frame.message_id.is_unique, model
    assert len(frame)==600 and set(frame.message_id)==set(audit11.message_id), model
    frame=frame.set_index('message_id').loc[audit11.message_id].reset_index()
    assert frame.pred_11.isin(LABELS11).all() and frame.pred_triage.isin(TRIAGE11).all(),model
    preds11[model]=frame
    print(f'Loaded {model}: {len(frame)} aligned predictions')

# Paired message-level bootstrap. Resamples the same IDs for each model (never train here).
rng11=np.random.default_rng(20261121)
boot11=[]
summary11=[]
rare11=[]
confusions11=[]
request_alerts11=[]
disagreements11=[]
for pop11, subset11 in [('all_600',np.ones(600,dtype=bool)),
                        ('relevant_504',audit11.disaster_relevance.eq('relevant').to_numpy())]:
    y11=audit11.humanitarian_label_11.to_numpy()[subset11]
    n11=int(subset11.sum())
    p11={model: df.pred_11.to_numpy()[subset11] for model,df in preds11.items()}
    for other11 in ('TFIDF_LinearSVC','Crisis_CT_M1'):
        a11=p11['VIRCON_full'];b11=p11[other11]
        delta11=float(f1_score(y11,a11,labels=LABELS11,average='macro',zero_division=0)
                      -f1_score(y11,b11,labels=LABELS11,average='macro',zero_division=0))
        draws11=np.empty(2000,dtype=float)
        for k11 in range(len(draws11)):
            sample11=rng11.integers(0,n11,size=n11)
            draws11[k11]=(f1_score(y11[sample11],a11[sample11],labels=LABELS11,average='macro',zero_division=0)
                         -f1_score(y11[sample11],b11[sample11],labels=LABELS11,average='macro',zero_division=0))
        lo11,hi11=np.quantile(draws11,[0.025,0.975])
        boot11.append({'population':pop11,'comparison':f'VIRCON_full minus {other11}',
                       'n_messages':n11,'macro_f1_difference':delta11,
                       'CI95_low':float(lo11),'CI95_high':float(hi11),
                       'CI_excludes_zero':bool(lo11>0 or hi11<0),'resamples':2000,
                       'sampling_unit':'message, conditional on one event'})
    mask3_11=subset11 & audit11.actionability_label.isin(TRIAGE11).to_numpy()
    y3_11=audit11.actionability_label.to_numpy()[mask3_11]
    for model11,df11 in preds11.items():
        prediction11=df11.pred_11.to_numpy()[subset11]
        pr11,re11,f111,s11=precision_recall_fscore_support(
            y11,prediction11,labels=LABELS11,zero_division=0)
        for lab11,precision11,recall11,f1class11,support11 in zip(LABELS11,pr11,re11,f111,s11):
            rare11.append({'population':pop11,'model':model11,'class':lab11,
                           'precision':float(precision11),'recall':float(recall11),
                           'f1':float(f1class11),'support':int(support11)})
        err11=pd.DataFrame({'truth':y11,'prediction':prediction11})
        err11=err11.loc[err11.truth.ne(err11.prediction)]
        pairs11=err11.groupby(['truth','prediction']).size().sort_values(ascending=False)
        for (truth11,pred11),count11 in pairs11.items():
            confusions11.append({'population':pop11,'model':model11,
                                 'true_class':truth11,'predicted_class':pred11,'count':int(count11)})
        p3_11=df11.pred_triage.to_numpy()[mask3_11]
        true_request11=y3_11=='request'
        request_prediction11=p3_11=='request'
        tp11=int(np.sum(true_request11 & request_prediction11))
        fp11=int(np.sum(~true_request11 & request_prediction11))
        fn11=int(np.sum(true_request11 & ~request_prediction11))
        summary11.append({'population':pop11,'model':model11,'n_11':n11,
                          'macro_f1_11':float(f1_score(y11,prediction11,labels=LABELS11,average='macro',zero_division=0)),
                          'n_actionability_evaluable':int(mask3_11.sum()),
                          'excluded_uncertain_actionability':int(subset11.sum()-mask3_11.sum()),
                          'request_TP':tp11,'request_FP':fp11,'request_FN':fn11,
                          'request_precision':tp11/(tp11+fp11) if tp11+fp11 else np.nan,
                          'request_recall':tp11/(tp11+fn11) if tp11+fn11 else np.nan})
        alert_mask11=mask3_11 & audit11.actionability_label.ne('request').to_numpy() & df11.pred_triage.eq('request').to_numpy()
        alerts11=pd.DataFrame({'message_id':audit11.message_id.to_numpy()[alert_mask11],
                'true_11':audit11.humanitarian_label_11.to_numpy()[alert_mask11],
                'pred_11':df11.pred_11.to_numpy()[alert_mask11],
                'true_actionability':audit11.actionability_label.to_numpy()[alert_mask11],
                'pred_actionability':df11.pred_triage.to_numpy()[alert_mask11]})
        alerts11.insert(0,'model',model11)
        alerts11.insert(0,'population',pop11)
        assert len(alerts11)==fp11,(model11,pop11,'false-alert count mismatch')
        request_alerts11.append(alerts11)
    full11=p11['VIRCON_full']
    for other11 in ('TFIDF_LinearSVC','Crisis_CT_M1'):
        right_alt11=(full11!=y11)&(p11[other11]==y11)
        right_full11=(full11==y11)&(p11[other11]!=y11)
        disagreements11.append({'population':pop11,'comparator':other11,
                    'VIRCON_wrong_comparator_correct':int(right_alt11.sum()),
                    'VIRCON_correct_comparator_wrong':int(right_full11.sum())})

# 'uncertain' is an explicit annotation state, not evidence that review is unfinished.
uncertain11=audit11.loc[audit11.actionability_label.eq('uncertain'),
                      ['message_id','disaster_relevance','humanitarian_label_11','actionability_label']]
uncertain11.to_csv(OUT11/'Excluded_Uncertain_Actionability_No_Text.csv',index=False)
pd.DataFrame(boot11).to_csv(OUT11/'Paired_Bootstrap_VIRCON_vs_Both_2000.csv',index=False)
pd.DataFrame(summary11).to_csv(OUT11/'Request_Alert_Audit.csv',index=False)
pd.DataFrame(rare11).to_csv(OUT11/'PerClass_Error_Analysis.csv',index=False)
pd.DataFrame(confusions11).to_csv(OUT11/'Misclassification_Pairs.csv',index=False)
pd.concat(request_alerts11,ignore_index=True).to_csv(OUT11/'False_Request_Alerts_No_Text.csv',index=False)
pd.DataFrame(disagreements11).to_csv(OUT11/'Paired_Model_Disagreements.csv',index=False)

# Optional evidence that the five cases were already human-confirmed.
confirm_files11=_candidate_files11('IJIES_Five_Label_Confirmed.csv')
confirmation11={'human_confirmation_file_found':bool(confirm_files11),
                'uncertain_actionability_count':int(len(uncertain11)),
                'annotation_changes_made_by_this_cell':0}
if confirm_files11:
    checks11=pd.read_csv(confirm_files11[0],dtype=str).fillna('')
    confirmation11['confirmation_rows']=int(len(checks11))
    if {'message_id','review_status','review_notes'}.issubset(checks11.columns):
        confirmation11['all_rows_confirmed']=bool(checks11.review_status.str.strip().str.lower().eq('confirmed').all())
        confirmation11['reviewed_ids_in_frozen']=bool(set(checks11.message_id).issubset(set(audit11.message_id)))
(OUT11/'Annotation_Audit_Notes.json').write_text(json.dumps(confirmation11,indent=2))

new_zip11=Path('/kaggle/working/IJIES_Post_Benchmark_Analysis_No_Text.zip')
with ZipFile(new_zip11,'w',compression=ZIP_DEFLATED) as z11:
    for file11 in sorted(OUT11.iterdir()):
        if file11.suffix in ('.csv','.json'):
            z11.write(file11,arcname=file11.name)
print('\nBOOTSTRAP RESULTS')
print(pd.DataFrame(boot11).to_string(index=False))
print('\nREQUEST ALERT AUDIT')
print(pd.DataFrame(summary11).to_string(index=False))
print('\nAnnotation audit:',confirmation11)
print('\nDOWNLOAD THE NEW ZIP FROM KAGGLE OUTPUT:',new_zip11)
