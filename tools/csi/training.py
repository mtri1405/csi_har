"""Explicit cascade: empty/present, then standing/walking on present only."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from csi.signal import (FEATURE_NAMES, MOTION_FEATURE_NAMES, ALL_FEATURE_NAMES,
                        LABELS, load_baseline)


def phase1_probability(bundle, X):
    X = np.asarray(X)
    return bundle['phase1'].predict_proba(X[:, :len(FEATURE_NAMES)])[
        :, list(bundle['phase1'].classes_).index(1)]


def predict_presence(bundle, X):
    return (phase1_probability(bundle, X) >= bundle['presence_threshold']).astype(int)


def predict(bundle, X):
    """Offline single-window cascade; phase2 is never called for empty rows."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != len(ALL_FEATURE_NAMES):
        raise ValueError('Expected 31 features: 13 presence + 18 motion; rebuild dataset')
    presence = predict_presence(bundle, X)
    result = np.zeros(len(X), dtype=int)
    mask = presence == 1
    if mask.any():
        result[mask] = bundle['phase2'].predict(X[mask, len(FEATURE_NAMES):]).astype(int)
    return result


def report(y, pred, labels, names):
    return dict(labels=labels, class_names=names,
                classification_report=classification_report(y, pred, labels=labels,
                    target_names=names, output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist())


def train(directory, out, trees=400, seed=42):
    directory, out = Path(directory), Path(out)
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if (metadata.get('pipeline_version') != 3 or
            metadata.get('feature_names') != ALL_FEATURE_NAMES):
        raise ValueError('Dataset version 3 required. Build dataset again: 13 presence + 18 motion.')
    frames = {s:pd.read_csv(directory/f'{s}_features.csv') for s in ('train','val','test')}
    seen = set()
    for split, df in frames.items():
        if set(df.label) != set(LABELS):
            raise ValueError(f'{split}: requires empty, standing and walking recordings')
        sessions = set(df.session_id)
        if sessions & seen:
            raise ValueError('Session leakage between splits')
        seen |= sessions
        if not np.isfinite(df[ALL_FEATURE_NAMES].to_numpy()).all():
            raise ValueError('Non-finite features')
        expected = (df.label != 'empty').astype(int).to_numpy()
        if not np.array_equal(df.phase1_y.to_numpy(), expected):
            raise ValueError('Phase 1 targets must map standing AND walking to present')
    df = frames['train']
    y = df.label.map(LABELS).to_numpy()
    X = df[ALL_FEATURE_NAMES].to_numpy()
    config = dict(n_estimators=trees, random_state=seed, class_weight='balanced', n_jobs=-1)
    # Shallow presence RF mirrors reference phase1/05_train_presence.py.
    p1 = RandomForestClassifier(**config, max_depth=5, min_samples_leaf=1)
    p2 = RandomForestClassifier(**config, max_depth=7, min_samples_leaf=1)
    p1.fit(X[:, :13], (y != 0).astype(int))
    p2.fit(X[y != 0, 13:], y[y != 0])
    counts = {label:int((df.label == label).sum()) for label in LABELS}
    training_data = dict(
        phase1=dict(empty=counts['empty'], present=counts['standing']+counts['walking'],
                    present_sources={k:counts[k] for k in ('standing','walking')},
                    features=13, sessions=int(df.session_id.nunique())),
        phase2=dict(standing=counts['standing'], walking=counts['walking'],
                    empty=0, features=18, sessions=int(df.loc[df.label!='empty','session_id'].nunique())),
        split_counts={s:frame.groupby('label').size().to_dict() for s,frame in frames.items()})
    bundle = dict(version=3, phase1=p1, phase2=p2, features=ALL_FEATURE_NAMES,
                  phase1_features=FEATURE_NAMES, phase2_features=MOTION_FEATURE_NAMES,
                  presence_threshold=.5, vote_windows=5, labels=LABELS,
                  metadata=metadata, training_data=training_data,
                  baseline=load_baseline(directory/'empty_baseline.npz', expected_layout=metadata['layout']))
    metrics = dict(version=3, training_data=training_data,
                   evaluation_protocol='Held-out sessions; single-window decisions, training baseline. Live calibration/voting are not included in these metrics.',
                   threshold=.5, baseline_source='TRAIN EMPTY recordings only')
    for split in ('val','test'):
        frame = frames[split]
        Xs = frame[ALL_FEATURE_NAMES].to_numpy()
        ys = frame.label.map(LABELS).to_numpy()
        mask = ys != 0
        metrics[split] = dict(
            phase1=report(mask.astype(int),predict_presence(bundle,Xs),[0,1],['empty','present']),
            phase2_true_presence=report(ys[mask],p2.predict(Xs[mask,13:]),[1,2],['standing','walking']),
            end_to_end=report(ys,predict(bundle,Xs),[0,1,2],['empty','standing','walking']),
            windows=len(ys), sessions=int(frame.session_id.nunique()))
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,out/'model.joblib')
    joblib.dump(dict(model=p1,feature_names=FEATURE_NAMES,labels={'empty':0,'present':1},
                    decision_threshold=.5,baseline=bundle['baseline'],metadata=metadata),out/'phase1_presence.joblib')
    joblib.dump(dict(model=p2,feature_names=MOTION_FEATURE_NAMES,labels={'standing':1,'walking':2},
                    requires='phase1 == present',metadata=metadata),out/'phase2_motion.joblib')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)
    return bundle


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--trees',type=int,default=400)
    args=parser.parse_args()
    if not 1 <= args.trees <= 10000:
        parser.error('trees must be 1..10000')
    train(args.features_dir,args.out,args.trees)
