"""Export every recording's reference processing stages for one receiver.

Uses the local reference project's 18 single-RX motion features directly.
Reference-style smoothing is applied within contiguous segments before windows.
This is an exploratory export, not the current production training pipeline.
"""
import argparse
import html
import importlib.util
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from csi.paths import DATA_ROOT, PROJECT_ROOT, REPORT_ROOT
from csi.protocol import unwrap_timestamps
from csi.signal import (raw_iq_to_amplitude, fieldwise_ltf_normalize, resample,
                        temporal_smooth, window_slices, extract_window_features,
                        FEATURE_NAMES)

FS = 25
GAP = .25
CARRIERS = [f'lltf_{i}' for i in range(51)] + [f'htltf_{i}' for i in range(56)]


def save_matrix(path, t, A, segment=None):
    df = pd.DataFrame(A, columns=CARRIERS)
    df.insert(0, 'time_s', t)
    if segment is not None:
        df.insert(0, 'segment', segment)
    df.to_csv(path, index=False, float_format='%.7g')


def load(path):
    df = pd.read_csv(path)
    if len(df) < 2:
        raise ValueError('Fewer than two packets')
    if df.channel.nunique() != 1:
        raise ValueError('Mixed channels')
    t = unwrap_timestamps(df.timestamp.to_numpy()) * 1e-6
    t -= t[0]
    valid, amp = [], []
    for i, r in enumerate(df.itertuples()):
        try:
            if r.len != 256 or r.sig != 1 or r.cwb or r.stbc or r.second_channel:
                continue
            amp.append(raw_iq_to_amplitude(r.data)); valid.append(i)
        except (ValueError, SyntaxError, TypeError):
            continue
    if len(valid) < 2:
        raise ValueError('Insufficient valid packets')
    t, A = t[valid], np.stack(amp)
    cuts = np.r_[0, np.flatnonzero(np.diff(t)>GAP)+1, len(t)]
    segments=[]
    for sid, (a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        if b-a < 2:
            continue
        rate=(b-a-1)/(t[b-1]-t[a])
        if rate>FS*1.3:
            raise ValueError('Source rate too high for interpolation without anti-alias filtering')
        grid=np.arange(t[a],t[b-1],1/FS)
        if not len(grid):
            continue
        R=resample(t[a:b],A[a:b],grid)
        N=fieldwise_ltf_normalize(R)
        S=temporal_smooth(N)
        segments.append(dict(id=sid,t=grid,resampled=R,normalized=N,smoothed=S,source_t=t[a:b]))
    if not segments:
        raise ValueError('No resampleable segments')
    profile=np.median(np.concatenate([s['smoothed'] for s in segments]),axis=0)
    return dict(path=path,label=path.parent.name,df=df,t=t,A=A,segments=segments,
                profile=profile,clean=bool((np.diff(t)<=GAP).all() and len(valid)==len(df)))


def plots(folder,d,ref,features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,9),layout='constrained')
    axes[0].plot(d['t'],d['A'].mean(axis=1),lw=.7)
    axes[0].set_ylabel('Raw mean amplitude')
    for seg in d['segments']:
        t=seg['t'];N=seg['normalized'];S=seg['smoothed']
        axes[1].plot(t,N.mean(axis=1),lw=.7,color='tab:blue')
        axes[2].plot(t,S.mean(axis=1),lw=.8,color='tab:orange')
        axes[3].plot(t,np.sqrt(np.mean((S-ref)**2,axis=1)),lw=.8,color='tab:red')
    axes[1].set_ylabel('Normalized mean');axes[2].set_ylabel('Smoothed mean')
    axes[3].set_ylabel('Baseline RMSE')
    for ax in axes:
        ax.set_xlabel('ESP time from recording start (s)');ax.grid(alpha=.2)
    fig.suptitle(d['path'].name+' | gaps are not interpolated')
    fig.savefig(folder/'signal_stages.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for seg in d['segments']:
        if len(seg['t']) < 2:
            continue
        extent=[seg['t'][0],seg['t'][-1],-.5,106.5]
        for ax,M,cmap,lo,hi in [(axes[0],seg['normalized'],'gray',.5,1.5),
                                (axes[1],seg['smoothed'],'gray',.5,1.5),
                                (axes[2],seg['smoothed']-ref,'RdBu_r',-.5,.5)]:
            ax.imshow(M.T,origin='lower',aspect='auto',extent=extent,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
    for ax,title in zip(axes,['Normalized gray picture [0.5, 1.5]',
                             'Savitzky-Golay gray picture [0.5, 1.5]',
                             'Smoothed minus empty baseline [-0.5, 0.5]']):
        ax.set_title(title);ax.set_xlim(0,d['t'][-1]);ax.set_facecolor('#ddd')
        ax.axhline(50.5,color='green',ls='--',lw=.8)
        ax.set_ylabel('Selected carrier');ax.set_xlabel('Time (s)')
    fig.suptitle(d['path'].name+' | fixed scales across files; gray gaps = no data')
    fig.savefig(folder/'heatmaps.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),layout='constrained')
    for sid,part in features.groupby('segment'):
        for ax,cols in zip(axes,[['temporal_std_mean','temporal_absdiff_mean'],
                                ['baseline_rmse','baseline_abs_mean'],
                                ['motion_bandpower_mean','motion_bandpower_p90']]):
            for col in cols:
                ax.plot(part.center_s,part[col],label=col if sid==features.segment.iloc[0] else None,lw=1)
    for ax in axes:
        ax.legend();ax.grid(alpha=.2);ax.set_xlabel('Window center (s)')
    fig.suptitle('Window features | '+d['path'].name)
    fig.savefig(folder/'feature_traces.png',dpi=140);plt.close(fig)


def run(out, room=None):
    out.mkdir(parents=True,exist_ok=True)
    source=PROJECT_ROOT/'ThamKhao/esp32-based-presence-and-motion-detection-using-wifi-csi-main/phase2/python/phase2_utils.py'
    spec=importlib.util.spec_from_file_location('reference_motion',source)
    motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
    recordings=[]
    errors=[]
    for path in sorted(DATA_ROOT.glob('*/*.csv')):
        try:
            recordings.append(load(path))
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            errors.append(dict(file=path.name,error=str(exc)))
            print(f'Rejected {path.name}: {exc}',flush=True)
    (out/'rejected_files.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    refs={d['path'].name:d['profile'] for d in recordings if d['label']=='empty' and d['clean']}
    if len(refs)<2:
        raise ValueError('At least two clean empty reference recordings required')
    all_features=[];overview=[];links=[]
    for d in recordings:
        name=d['path'].name
        reference_names=[n for n in refs if d['label']!='empty' or n!=name]
        ref=np.median(np.stack([refs[n] for n in reference_names]),axis=0)
        folder=out/d['label']/d['path'].stem;folder.mkdir(parents=True,exist_ok=True)
        save_matrix(folder/'01_raw_amplitude.csv',d['t'],d['A'])
        t=np.concatenate([s['t'] for s in d['segments']])
        ids=np.concatenate([np.full(len(s['t']),s['id']) for s in d['segments']])
        for filename,key in [('02_resampled.csv','resampled'),('03_normalized.csv','normalized'),('04_smoothed.csv','smoothed')]:
            save_matrix(folder/filename,t,np.concatenate([s[key] for s in d['segments']]),ids)
        pd.DataFrame(dict(carrier=CARRIERS,baseline=ref)).to_csv(folder/'baseline_used.csv',index=False)
        rows=[];windows=[]
        for s in d['segments']:
            for sl in window_slices(len(s['t']),FS,2,.5):
                W=s['smoothed'][sl]
                f=extract_window_features(W,ref)
                mf=motion._receiver_motion_features(W,FS)
                rows.append(dict(segment=s['id'],start_s=float(s['t'][sl][0]),
                    end_s=float(s['t'][sl][-1]),center_s=float(s['t'][sl].mean()),
                    observed_packets=int(((s['source_t']>=s['t'][sl][0]) & (s['source_t']<=s['t'][sl][-1])).sum()),
                    **dict(zip(FEATURE_NAMES,map(float,f))),
                    **dict(zip(['motion_'+n for n in motion.RX_MOTION_FEATURE_NAMES],map(float,mf)))))
                windows.append(W)
        fdf=pd.DataFrame(rows)
        if fdf.empty:
            raise ValueError(f'{name}: no complete windows')
        numeric=fdf.to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f'{name}: nonfinite features')
        fdf.to_csv(folder/'05_window_features.csv',index=False)
        np.savez_compressed(folder/'processed_windows.npz',X=np.stack(windows),
                            baseline=ref,center_s=fdf.center_s.to_numpy(),segment=fdf.segment.to_numpy())
        fdf.drop(columns=['segment','start_s','end_s','center_s','observed_packets']).agg(['mean','median','std','min','max']).T.to_csv(folder/'06_feature_summary.csv')
        meta=dict(file=name,label=d['label'],raw_packets=len(d['df']),valid_packets=len(d['A']),
                  duration_s=float(d['t'][-1]),fs_hz=FS,window_samples=50,requested_step_s=.5,
                  effective_step_s=.48,windows=len(fdf),segments=len(d['segments']),
                  gaps_over_250ms=int((np.diff(d['t'])>GAP).sum()),
                  baseline_recordings=reference_names,
                  processing='IQ amplitude -> 107 carriers -> linear resample -> median per LTF -> Savitzky-Golay 5/poly2 per segment -> windows -> 13 presence + 18 motion features',
                  differences_from_training='Reference-style segment smoothing, not production window-local smoothing. No model fitted.',
                  limitations='Exploratory empty baseline; not a train/test split. No fan-specific filtering. Interpolation cannot recover missing observations.',
                  acquisition=dict(room=room,source='Room supplied by user; board geometry, fan state and walking direction must be confirmed for each new setup.'))
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        plots(folder,d,ref,fdf)
        files=['01_raw_amplitude.csv','02_resampled.csv','03_normalized.csv','04_smoothed.csv',
               '05_window_features.csv','06_feature_summary.csv','processed_windows.npz','metadata.json','baseline_used.csv']
        page='<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}</style>'
        page+=f'<h1>{html.escape(name)}</h1><p>{len(fdf)} windows; 13 presence + 18 motion features. Exploratory only.</p>'
        page+='<p>Fixed heatmap scales across recordings. Missing-data gaps are not interpolated. Baseline excludes this recording when empty.</p>'
        page+='<ul>'+''.join(f'<li><a href="{f}">{f}</a></li>' for f in files)+'</ul>'
        page+=''.join(f'<h2>{f}</h2><img src="{f}">' for f in ['signal_stages.png','heatmaps.png','feature_traces.png'])
        (folder/'index.html').write_text(page,encoding='utf-8')
        relative=(folder/'index.html').relative_to(out).as_posix()
        links.append(f'<li><a href="{relative}">{html.escape(name)}</a> — {len(fdf)} windows</li>')
        overview.append({k:meta[k] for k in ['file','label','raw_packets','duration_s','windows','segments','gaps_over_250ms']})
        all_features.append(fdf.assign(file=name,label=d['label']))
        print(f'Exported {name}: {len(fdf)} windows',flush=True)
    pd.DataFrame(overview).to_csv(out/'overview.csv',index=False)
    pd.concat(all_features,ignore_index=True).to_csv(out/'all_window_features.csv',index=False)
    page='<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1000px;margin:30px auto}li{margin:12px}</style><h1>CSI processing: per-recording exports</h1><p>Single RX. Exploratory comparison; no classification accuracy. Raw CSV files unchanged.</p><p>Stages and 31 features follow the local reference project. Smoothed segment exports differ from the production window-local smoothing pipeline. No filter has been fitted to the ceiling fan.</p><ul>'+''.join(links)+'</ul>'
    (out/'index.html').write_text(page,encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print(f'Done: {len(recordings)} files, {sum(len(f) for f in all_features)} windows; {archive}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPORT_ROOT/'per_file_processing_20261117')
    parser.add_argument('--room',default=None)
    args=parser.parse_args()
    run(args.out,room=args.room)
