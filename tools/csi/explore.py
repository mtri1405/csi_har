"""Descriptive audit and feature extraction; no model fitting or accuracy claims.

Split at >250ms gaps before interpolation. The training pipeline is unchanged.
Empty feature references use leave-one-recording-out baselines. Person windows
use the median baseline from the seven full-recording-valid empty recordings.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from csi.paths import DATA_ROOT, REPORT_ROOT
from csi.protocol import unwrap_timestamps
from csi.signal import (raw_iq_to_amplitude, fieldwise_ltf_normalize, resample,
                        temporal_smooth, window_slices, extract_window_features,
                        FEATURE_NAMES, MAX_INTERP_GAP_S)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    series, audit = {}, []
    for path in sorted(DATA_ROOT.glob('*/*.csv')):
        df = pd.read_csv(path)
        if df.empty:
            audit.append(dict(file=path.name, label=path.parent.name, rows=0,
                              error='No packets'))
            continue
        label = path.parent.name
        t = unwrap_timestamps(pd.to_numeric(df.timestamp).to_numpy()) * 1e-6
        t -= t[0]
        dt = np.diff(t)
        arrays, valid = [], []
        for i, row in enumerate(df.itertuples()):
            try:
                if (row.len != 256 or row.sig != 1 or row.cwb != 0 or
                        row.stbc != 0 or row.second_channel != 0):
                    continue
                arrays.append(raw_iq_to_amplitude(row.data))
                valid.append(i)
            except (ValueError, TypeError, SyntaxError):
                continue
        channels = df.channel.nunique()
        row = dict(file=path.name, label=label, rows=len(df), valid_rows=len(valid),
                   rejected_rows=len(df)-len(valid), esp_duration_s=float(t[-1]),
                   rate_hz=float((len(t)-1)/t[-1]),
                   median_interval_ms=float(np.median(dt)*1000),
                   max_gap_s=float(dt.max()), gaps_over_250ms=int((dt>.25).sum()),
                   seq_missing=int(np.maximum(0, np.diff(df.seq)-1).sum()),
                   firmware_drop_delta=int(df.dropped.iloc[-1]-df.dropped.iloc[0]),
                   rssi_mean=float(df.rssi.mean()), rssi_std=float(df.rssi.std()),
                   channels=channels, mcs_values=','.join(map(str, sorted(df.mcs.unique()))),
                   full_recording_valid=bool((dt <= .25).all() and len(valid)==len(df) and channels==1))
        if 'pc_elapsed_s' in df:
            pc = df.pc_elapsed_s.to_numpy()
            row.update(pc_duration_s=float(pc[-1]-pc[0]), first_pc_s=float(pc[0]),
                       last_pc_s=float(pc[-1]), clock_difference_s=float(t[-1]-(pc[-1]-pc[0])))
        audit.append(row)
        if len(valid) < 2 or channels != 1:
            continue
        tv = t[valid]
        A = np.stack(arrays)
        cuts = np.r_[0, np.flatnonzero(np.diff(tv)>MAX_INTERP_GAP_S)+1, len(tv)]
        segments=[]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            if stop-start < 2 or tv[stop-1]-tv[start] < 1.96:
                continue
            grid=np.arange(tv[start],tv[stop-1],1/25)
            B=fieldwise_ltf_normalize(resample(tv[start:stop],A[start:stop],grid))
            segments.append((grid, B, tv[start:stop]))
        series[path.name]=dict(label=label, segments=segments,
                              profile=np.median(fieldwise_ltf_normalize(A),axis=0),
                              full_valid=row['full_recording_valid'])
    adf=pd.DataFrame(audit)
    adf.to_csv(out/'recording_audit.csv',index=False)
    references={name:s['profile'] for name,s in series.items()
                if s['label']=='empty' and s['full_valid']}
    if len(references)<2:
        raise ValueError('Need >=2 clean empty recordings for exploratory baseline')
    baseline=np.median(np.stack(list(references.values())),axis=0)
    np.savez_compressed(out/'exploratory_baseline.npz', amplitude=baseline)
    features=[]
    for name,s in series.items():
        ref_profiles=[v for k,v in references.items() if k != name]
        ref=np.median(np.stack(ref_profiles),axis=0) if s['label']=='empty' else baseline
        for segment_id,(grid,B,raw_t) in enumerate(s['segments']):
            for sl in window_slices(len(grid),25,2,.5):
                W=temporal_smooth(B[sl])
                f=extract_window_features(W,ref)
                observed=int(((raw_t>=grid[sl][0]) & (raw_t<=grid[sl][-1])).sum())
                features.append(dict(file=name,label=s['label'],segment=segment_id,
                    center_s=float(grid[sl].mean()),observed_packets=observed,
                    **dict(zip(FEATURE_NAMES,map(float,f)))))
    fdf=pd.DataFrame(features)
    fdf.to_csv(out/'window_features.csv',index=False)
    med=fdf.groupby(['label','file'])[FEATURE_NAMES].median()
    med.to_csv(out/'session_feature_medians.csv')
    fdf.groupby('label')[FEATURE_NAMES].agg(['median','min','max']).to_csv(out/'label_feature_summary.csv')
    counts=fdf.groupby('file').size()
    adf['extracted_windows']=adf.file.map(counts).fillna(0).astype(int)
    adf.to_csv(out/'recording_audit.csv',index=False)
    plot(out, med, fdf, series, baseline)
    info=dict(files=len(adf),clean_full_recordings=int(adf.full_recording_valid.fillna(False).sum()),
              windows=int(len(fdf)),windows_by_label=fdf.groupby('label').size().to_dict(),
              baseline_recordings=list(references),fs_hz=25,window_s=2,step_s=.5,effective_step_s=12/25,
              gap_policy='Split >250ms gaps; discard segments shorter than a full window.',
              baseline_policy='Clean empty session median profiles; empty references exclude own recording.',
              limitations=['Exploratory only; no train/val/test or model accuracy.',
                           'Overlapping windows are correlated; compare session medians.',
                           'Person recordings are later in time than empty; class and time can be confounded.',
                           'Resampling to 25Hz does not recover missing observations.',
                           'Left/right are filename annotations; physical placement not verified.'])
    (out/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    report=['# Exploratory CSI audit','',json.dumps(info,indent=2),'',
            '## Recording audit','',adf.to_csv(index=False),'',
            '## Median features per recording','',med.to_csv(),'',
            'No classification accuracy was computed. Raw recordings were not modified.']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)
    print(adf[['file','rate_hz','max_gap_s','clock_difference_s','extracted_windows']].to_string(index=False),flush=True)
    print(med[['temporal_std_mean','temporal_absdiff_mean','baseline_abs_mean','baseline_rmse','baseline_corr']].to_string(),flush=True)


def plot(out,med,features,series,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'empty':'#2378ba','standing':'#e19a23','walking':'#c84949'}
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(x,y) in zip(axes,[('baseline_rmse','temporal_std_mean'),
                             ('baseline_abs_mean','temporal_absdiff_mean'),
                             ('baseline_corr','temporal_std_mean')]):
        for label in colors:
            part=med.loc[label]
            ax.scatter(part[x],part[y],label=label,c=colors[label],s=65)
            for name,r in part.iterrows():
                short=('E'+str(list(part.index).index(name)+1) if label=='empty'
                       else ('S' if label=='standing' else 'W')+('L' if '_left_' in name else 'R'))
                ax.annotate(short,(r[x],r[y]),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('One point per recording: median of window features (exploratory)')
    fig.savefig(out/'session_features.png',dpi=170);plt.close(fig)
    # Same scale for each heatmap: normalization and baseline shared across labels.
    names=[next(n for n,s in series.items() if s['label']==label and len(s['segments']))
           for label in ('empty','standing','walking')]
    fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    for ax,name in zip(axes,names):
        s=series[name];grid,B,_=max(s['segments'],key=lambda seg:len(seg[0]))
        ref=np.median(np.stack([v['profile'] for n,v in series.items()
                               if v['label']=='empty' and v['full_valid'] and n!=name]),axis=0)
        im=ax.imshow((B-ref).T,aspect='auto',origin='lower',extent=[grid[0],grid[-1],0,107],
                     cmap='RdBu_r',vmin=-.5,vmax=.5)
        ax.axhline(51,color='black',ls='--',lw=.6)
        ax.set_title(name+' (longest valid segment)');ax.set_xlabel('ESP time from recording start (s)');ax.set_ylabel('Selected carrier')
    fig.colorbar(im,ax=axes,label='Normalized amplitude minus empty reference')
    fig.savefig(out/'comparison_heatmaps.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=REPORT_ROOT/'exploratory_20261117')
    run(p.parse_args().out)
