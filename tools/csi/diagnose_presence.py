"""Read-only model diagnosis: per-session errors, train ranges and ablations.

Diagnostic models use TRAIN only and are evaluated on VAL only. Saved deployment
models, thresholds, splits and baselines are never overwritten by this script.
"""
import argparse
import html
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from csi.paths import REPORT_ROOT
from csi.signal import FEATURE_NAMES


def run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    b=joblib.load(REPORT_ROOT/'two_phase_rf/model.joblib')
    model=b['phase1'];model.n_jobs=1
    df=pd.read_csv(REPORT_ROOT/'features/all_features.csv')
    manifest=pd.read_csv(REPORT_ROOT/'web_manifest.csv')
    df['p_present']=model.predict_proba(df[FEATURE_NAMES].to_numpy())[:,list(model.classes_).index(1)]
    df['pred_present']=df.p_present>=b['presence_threshold']
    df['true_present']=df.label!='empty'
    df['incorrect']=df.pred_present!=df.true_present
    df.to_csv(out/'window_predictions.csv',index=False)
    summary=df.groupby(['split','label','session_id']).agg(
        windows=('label','size'),predicted_present=('pred_present','sum'),errors=('incorrect','sum'),
        p_present_min=('p_present','min'),p_present_median=('p_present','median'),p_present_max=('p_present','max')).reset_index()
    summary['recall']=1-summary.errors/summary.windows
    summary=summary.merge(manifest[['session_id','rx_csv']],on='session_id')
    summary.to_csv(out/'session_errors.csv',index=False)
    med=df.groupby(['split','label','session_id'])[FEATURE_NAMES].median().reset_index()
    med.to_csv(out/'session_feature_medians.csv',index=False)
    importance=pd.Series(model.feature_importances_,index=FEATURE_NAMES).sort_values(ascending=False)
    importance.rename('impurity_importance').to_csv(out/'feature_importance.csv')
    train=df[df.split=='train'];val=df[df.split=='val']
    empty=train[train.label=='empty']
    rows=[]
    for session,part in val[val.label=='empty'].groupby('session_id'):
        for feature in FEATURE_NAMES:
            lo,hi=empty[feature].min(),empty[feature].max()
            rows.append(dict(session_id=session,feature=feature,train_empty_min=lo,train_empty_max=hi,
                             val_median=part[feature].median(),val_min=part[feature].min(),val_max=part[feature].max(),
                             outside_train_empty_fraction=float(((part[feature]<lo)|(part[feature]>hi)).mean())))
    pd.DataFrame(rows).to_csv(out/'validation_empty_vs_training_range.csv',index=False)
    groups=dict(full_presence=FEATURE_NAMES,
                profile_only=FEATURE_NAMES[:3],baseline_only=[n for n in FEATURE_NAMES if n.startswith('baseline_')],
                temporal_only=FEATURE_NAMES[3:8],profile_and_correlation=FEATURE_NAMES[:3]+['baseline_corr'])
    ablations=[]
    for name,names in groups.items():
        if name=='full_presence':
            prob=model.predict_proba(val[names].to_numpy())[:,1]
        else:
            diagnostic=RandomForestClassifier(n_estimators=400,max_depth=5,min_samples_leaf=1,
                                              class_weight='balanced',random_state=42,n_jobs=1)
            diagnostic.fit(train[names].to_numpy(),train.true_present.astype(int))
            prob=diagnostic.predict_proba(val[names].to_numpy())[:,list(diagnostic.classes_).index(1)]
        pred=prob>=b['presence_threshold'];y=val.true_present.astype(int).to_numpy()
        cm=confusion_matrix(y,pred,labels=[0,1])
        ablations.append(dict(features=name,feature_count=len(names),validation_accuracy=float(accuracy_score(y,pred)),
                              empty_recall=float(recall_score(y,pred,pos_label=0)),
                              present_recall=float(recall_score(y,pred,pos_label=1)),
                              false_present=int(cm[0,1]),false_empty=int(cm[1,0])))
    pd.DataFrame(ablations).to_csv(out/'diagnostic_ablations_val_only.csv',index=False)
    # Counterfactual is a sensitivity probe, NOT a physically realized CSI signal.
    controls=[]
    base_names=[n for n in FEATURE_NAMES if n.startswith('baseline_')]
    for session,part in val[val.label=='empty'].groupby('session_id'):
        X=part[FEATURE_NAMES].copy()
        X[base_names]=np.tile(empty[base_names].median().to_numpy(),(len(X),1))
        prob=model.predict_proba(X.to_numpy())[:,1]
        controls.append(dict(session_id=session,original_present=int(part.pred_present.sum()),
                             counterfactual_present=int((prob>=.5).sum()),
                             original_pmedian=float(part.p_present.median()),
                             counterfactual_pmedian=float(np.median(prob))))
    pd.DataFrame(controls).to_csv(out/'baseline_sensitivity_not_a_fix.csv',index=False)
    # Compare the older descriptive reference with the actual deployed train baseline.
    exploratory=REPORT_ROOT/'bedroom_review_20261118/session_medians.csv'
    baseline_comparison=[]
    if exploratory.exists():
        prior=pd.read_csv(exploratory)
        for r in med.itertuples():
            path=Path(manifest.loc[manifest.session_id==r.session_id,'rx_csv'].iloc[0])
            old=prior[prior.file==path.name]
            if len(old):
                baseline_comparison.append(dict(session_id=r.session_id,label=r.label,split=r.split,
                    current_train_baseline_rmse=r.baseline_rmse,
                    exploratory_baseline_rmse=float(old.baseline_rmse.iloc[0])))
    pd.DataFrame(baseline_comparison).to_csv(out/'exploratory_vs_training_baseline.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    colors={'empty':'#287ab8','standing':'#dd9b27','walking':'#ba4747'}
    for ax,split in zip(axes[:2],['train','val']):
        part=df[df.split==split]
        for label,color in colors.items():
            p=part[part.label==label]
            ax.scatter(p.baseline_rmse,p.mean_profile_std,c=color,s=9,alpha=.4,label=label)
        bad=part[part.incorrect]
        ax.scatter(bad.baseline_rmse,bad.mean_profile_std,c='red',marker='x',s=22,label='wrong')
        for r in med[(med.split==split)&(med.label=='empty')].itertuples():
            ax.annotate(r.session_id,(r.baseline_rmse,r.mean_profile_std),fontsize=8)
        ax.axvline(empty.baseline_rmse.max(),color='gray',ls='--',label='max TRAIN empty RMSE')
        ax.set_title(split.upper()+' | actual model features');ax.set_xlabel('baseline_rmse');ax.set_ylabel('mean_profile_std');ax.legend(fontsize=8);ax.grid(alpha=.2)
    sv=summary[summary.split=='val']
    for i,r in enumerate(sv.itertuples()):
        axes[2].plot([r.p_present_min,r.p_present_max],[i,i],c=colors[r.label],lw=3)
        axes[2].scatter([r.p_present_median],[i],c=colors[r.label],s=40)
    axes[2].set_yticks(range(len(sv)),sv.session_id);axes[2].axvline(.5,c='red',ls='--')
    axes[2].set_xlim(-.02,1.02);axes[2].set_xlabel('P(present)');axes[2].set_title('VAL probabilities | line=min/max, dot=median');axes[2].grid(alpha=.2)
    fig.savefig(out/'why_empty_is_misclassified.png',dpi=160);plt.close(fig)
    info=dict(baseline_importance=float(importance[base_names].sum()),
              training_empty_sessions=sorted(empty.session_id.unique()),threshold=b['presence_threshold'],
              ablations=ablations,baseline_sensitivity=controls,
              limitation='Exploratory diagnosis. No production model/threshold changed. Diagnostic models fit TRAIN only and score VAL only. Counterfactual features need not correspond to physically possible CSI.')
    (out/'diagnosis.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    table=summary[summary.split=='val'].to_html(index=False,escape=True,float_format=lambda x:f'{x:.4f}')
    page='<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1400px;margin:25px auto}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Phase 1: chẩn đoán recall EMPTY</h1>'
    page+='<p>RF được kiểm tra ở ngưỡng 0,5, chỉ dùng 13 đặc trưng presence. Không sửa model hoặc ngưỡng đang dùng.</p><p>Bảng dưới là từng phiên validation. Recall thấp tập trung ở empty_002 và empty_008.</p>'+table
    page+='<h2>Đặc trưng và xác suất</h2><img src="why_empty_is_misclassified.png"><h2>Thử nghiệm bỏ nhóm đặc trưng</h2><p>Chỉ fit TRAIN, chỉ đánh giá VAL. Đây là chẩn đoán, chưa phải lựa chọn model cuối.</p>'+pd.DataFrame(ablations).to_html(index=False,float_format=lambda x:f'{x:.4f}')
    page+='<h2>Ảnh hưởng baseline</h2><p>Thay nhóm baseline bằng trung vị train-empty để kiểm tra độ nhạy model; đây không phải phép sửa dữ liệu hợp lệ.</p>'+pd.DataFrame(controls).to_html(index=False,float_format=lambda x:f'{x:.4f}')
    for name in ['session_errors.csv','session_feature_medians.csv','feature_importance.csv',
                 'validation_empty_vs_training_range.csv','exploratory_vs_training_baseline.csv',
                 'diagnostic_ablations_val_only.csv','window_predictions.csv']:
        page+=f'<p><a href="{name}">{name}</a></p>'
    (out/'index.html').write_text(page,encoding='utf-8')
    print(summary[['split','label','session_id','windows','errors','recall','p_present_median']].to_string(index=False),flush=True)
    print('BASELINE_IMPORTANCE',info['baseline_importance'],flush=True)
    print(pd.DataFrame(ablations).to_string(index=False),flush=True)
    print(pd.DataFrame(controls).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPORT_ROOT/'presence_diagnosis_20261118')
    run(parser.parse_args().out)
