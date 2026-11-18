"""Independently check exported formulas and build a recording-level review.

No model training or label verification. Class differences are descriptive.
"""
import argparse
import ast
import html
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from csi.paths import DATA_ROOT
from csi.protocol import unwrap_timestamps
from csi.signal import FEATURE_NAMES, temporal_smooth, extract_window_features


def independent_features(window, reference):
    W=window.astype(np.float64);b=reference.astype(np.float64)
    m=np.mean(W,axis=0);sd=np.std(W,axis=0);d=W[1:]-W[:-1];ad=np.abs(d)
    delta=np.abs(m-b)
    corr=np.corrcoef(m,b)[0,1] if np.std(m)>1e-6 and np.std(b)>1e-6 else 0.
    p=[np.mean(m),np.std(m),np.median(m),np.mean(sd),np.median(sd),np.max(sd),
       np.mean(ad),np.std(d),np.mean(delta),np.median(delta),np.max(delta),
       np.sqrt(np.mean((m-b)**2)),corr]
    absdiff=np.mean(ad,axis=0);rms=np.sqrt(np.mean(d*d,axis=0));ranges=np.ptp(W,axis=0)
    packet=np.sqrt(np.mean(d*d,axis=1))
    spec=np.fft.rfft(W-np.mean(W,axis=0),axis=0)
    freq=np.fft.rfftfreq(len(W),1/25)
    power=np.abs(spec)**2/len(W)
    band=np.log1p(np.mean(power[(freq>=.5)&(freq<=5)],axis=0))
    q=lambda a:np.percentile(a,90)
    motion=[np.mean(sd),np.median(sd),q(sd),np.max(sd),
            np.mean(absdiff),np.median(absdiff),q(absdiff),np.max(absdiff),
            np.mean(rms),q(rms),np.mean(ranges),q(ranges),
            np.mean(packet),np.std(packet),q(packet),np.max(packet),np.mean(band),q(band)]
    return np.asarray(p+motion)


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_features=pd.read_csv(root/'all_window_features.csv')
    feature_names=FEATURE_NAMES+[n for n in all_features.columns if n.startswith('motion_')]
    checks=[];audits=[];matrices=[];production=[]
    counters={label:0 for label in ('empty','standing','walking')}
    for file,features in all_features.groupby('file',sort=True):
        label=features.label.iloc[0];counters[label]+=1
        identifier={'empty':'E','standing':'S','walking':'W'}[label]+str(counters[label])
        folder=root/label/Path(file).stem
        raw=pd.read_csv(DATA_ROOT/label/file)
        exported_raw=pd.read_csv(folder/'01_raw_amplitude.csv')
        iq=np.asarray([ast.literal_eval(v) for v in raw.data],dtype=np.float64)
        amplitude=np.hypot(iq[:,0::2],iq[:,1::2])
        idx=np.r_[np.arange(2,27),np.arange(38,64),64+np.arange(1,29),64+np.arange(36,64)]
        np.testing.assert_allclose(exported_raw.iloc[:,1:].to_numpy(),amplitude[:,idx],rtol=1e-6,atol=1e-4)
        t=unwrap_timestamps(raw.timestamp.to_numpy())*1e-6;t-=t[0]
        pc=raw.pc_elapsed_s.to_numpy()
        norm=pd.read_csv(folder/'03_normalized.csv')
        smoothed=pd.read_csv(folder/'04_smoothed.csv')
        columns=[c for c in norm if c.startswith(('lltf_','htltf_'))]
        N=norm[columns].to_numpy();S=smoothed[columns].to_numpy()
        with np.load(folder/'processed_windows.npz') as z:
            X=z['X'];baseline=z['baseline']
        recomputed=np.vstack([independent_features(W,baseline) for W in X])
        saved=features[feature_names].to_numpy()
        difference=np.abs(saved-recomputed)
        passed=np.allclose(saved,recomputed,atol=2e-6,rtol=2e-4)
        # Compare the production window-local smoother with reference segment smoothing.
        local=[]
        for r in features.itertuples():
            part=norm.loc[(norm.segment==r.segment)&(norm.time_s>=r.start_s-1e-5)&(norm.time_s<=r.end_s+1e-5),columns]
            if len(part)!=50:
                raise AssertionError(f'{file}: incorrect window slice: {len(part)}')
            local.append(extract_window_features(temporal_smooth(part.to_numpy()),baseline))
        local=np.vstack(local)
        local_frame=pd.DataFrame(local,columns=FEATURE_NAMES)
        local_frame.insert(0,'center_s',features.center_s.to_numpy())
        local_frame.insert(0,'label',label);local_frame.insert(0,'file',file)
        production.append(local_frame)
        source_rate=(len(raw)-1)/t[-1]
        summary=dict(file=file,id=identifier,label=label,packets=len(raw),
                     valid_packets=len(pd.read_csv(folder/'01_raw_amplitude.csv')),
                     duration_esp_s=float(t[-1]),duration_pc_s=float(pc[-1]-pc[0]),
                     first_packet_pc_s=float(pc[0]),last_packet_pc_s=float(pc[-1]),
                     clock_difference_ms=float((t[-1]-pc[-1]+pc[0])*1000),
                     rate_hz=float(source_rate),max_gap_ms=float(np.diff(t).max()*1000),
                     seq_missing=int(np.maximum(0,np.diff(raw.seq)-1).sum()),
                     firmware_drop_delta=int(raw.dropped.iloc[-1]-raw.dropped.iloc[0]),
                     rssi_mean=float(raw.rssi.mean()),rssi_std=float(raw.rssi.std()),
                     mcs_values=','.join(map(str,sorted(raw.mcs.unique()))),
                     channels=','.join(map(str,sorted(raw.channel.unique()))),
                     windows=len(features),feature_formulas_pass=bool(passed),
                     max_feature_absolute_error=float(difference.max()),
                     max_ltf_median_error=float(max(np.abs(np.median(N[:,:51],axis=1)-1).max(),np.abs(np.median(N[:,51:],axis=1)-1).max())),
                     normalized_heatmap_clipped_fraction=float(((N<.5)|(N>1.5)).mean()),
                     segment_vs_local_max_difference=float(np.max(np.abs(local-saved[:,:13]))))
        if not passed or not np.isfinite(saved).all() or X.shape!=(len(features),50,107):
            raise AssertionError(f'{file}: feature verification failed')
        if label=='empty' and file in json.loads((folder/'metadata.json').read_text())['baseline_recordings']:
            raise AssertionError(f'{file}: own recording in empty reference')
        audits.append(summary)
        checks.append(dict(file=file,windows=len(features),feature_count=len(feature_names),
                           passed=bool(passed),max_absolute_error=float(difference.max())))
        matrices.append((identifier,file,label,smoothed.time_s.to_numpy(),S,S-baseline))
        print(f'Checked {identifier}: {file}, {len(features)} windows, max error {difference.max():.3g}',flush=True)
    audit=pd.DataFrame(audits)
    pd.concat(production,ignore_index=True).to_csv(root/'window_local_presence_features.csv',index=False)
    audit.to_csv(root/'quality_and_verification.csv',index=False)
    (root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    med=all_features.groupby(['label','file'])[feature_names].median().reset_index()
    med=med.merge(audit[['file','id']],on='file')
    med.to_csv(root/'session_medians.csv',index=False)
    med.groupby('label')[feature_names].agg(['min','median','max']).to_csv(root/'class_session_ranges.csv')
    # Class contrast uses one point per recording rather than correlated windows.
    contrast=[]
    for feature in feature_names:
        a=med.loc[med.label=='empty',feature].to_numpy()
        b=med.loc[med.label!='empty',feature].to_numpy()
        s=med.loc[med.label=='standing',feature].to_numpy()
        w=med.loc[med.label=='walking',feature].to_numpy()
        contrast.append(dict(feature=feature,empty_min=float(a.min()),empty_max=float(a.max()),
                             present_min=float(b.min()),present_max=float(b.max()),
                             empty_present_ranges_disjoint=bool(a.max()<b.min() or b.max()<a.min()),
                             standing_min=float(s.min()),standing_max=float(s.max()),
                             walking_min=float(w.min()),walking_max=float(w.max()),
                             standing_walking_ranges_disjoint=bool(s.max()<w.min() or w.max()<s.min())))
    pd.DataFrame(contrast).to_csv(root/'descriptive_class_contrasts.csv',index=False)
    for delta,filename in [(False,'all_smoothed_heatmaps.png'),(True,'all_baseline_heatmaps.png')]:
        nrows=max(1,(len(matrices)+3)//4)
        fig,axes=plt.subplots(nrows,4,figsize=(17,3*nrows),layout='constrained',squeeze=False)
        for ax,item in zip(axes.flat,matrices):
            identifier,file,label,t,S,D=item
            im=ax.imshow((D if delta else S).T,origin='lower',aspect='auto',
                         extent=[t[0],t[-1],0,107],cmap='RdBu_r' if delta else 'gray',
                         vmin=-.5 if delta else .5,vmax=.5 if delta else 1.5)
            ax.set_title(f'{identifier}: {label}',fontsize=11)
            ax.axhline(51,color='green',lw=.6,ls='--');ax.set_xlabel('Time (s)');ax.set_ylabel('Carrier')
        for ax in list(axes.flat)[len(matrices):]:ax.set_visible(False)
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.5,label='Relative amplitude difference' if delta else 'Relative amplitude')
        fig.suptitle('Bedroom recordings | fixed color scales | descriptive only',fontsize=15)
        fig.savefig(root/filename,dpi=130);plt.close(fig)
    colors={'empty':'#2677b5','standing':'#df9b29','walking':'#c74646'}
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    selected=['temporal_std_mean','temporal_absdiff_mean','baseline_rmse','baseline_corr',
              'motion_bandpower_mean','motion_packet_change_mean']
    for ax,feature in zip(axes.flat,selected):
        for i,label in enumerate(colors):
            part=med[med.label==label]
            x=np.linspace(i-.15,i+.15,len(part))
            ax.scatter(x,part[feature],color=colors[label],s=45)
            for xx,(_,r) in zip(x,part.iterrows()):ax.annotate(r.id,(xx,r[feature]),fontsize=8,xytext=(3,2),textcoords='offset points')
        ax.set_xticks([0,1,2],list(colors));ax.set_title(feature);ax.grid(axis='y',alpha=.2)
    fig.suptitle('One dot per recording: median window feature; no model accuracy')
    fig.savefig(root/'recording_feature_comparison.png',dpi=150);plt.close(fig)
    blocks=[]
    for r in audit.itertuples():
        relative=f'{r.label}/{Path(r.file).stem}/index.html'
        blocks.append(f'<tr><td>{r.id}</td><td><a href="{relative}">{html.escape(r.file)}</a></td><td>{r.packets}</td><td>{r.duration_esp_s:.2f}</td><td>{r.rate_hz:.2f}</td><td>{r.max_gap_ms:.1f}</td><td>{r.seq_missing}</td><td>PASS</td></tr>')
    page='''<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:15px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:14px}a{color:#1768a6}</style>
<h1>Phòng ngủ: kiểm tra CSI và đặc trưng</h1>
<p>20 recording. Mỗi file có heatmap, tín hiệu qua từng bước và 31 đặc trưng. PASS chỉ xác nhận công thức tính; không xác nhận nhãn thực tế hoặc độ chính xác nhận diện.</p>
<p>Baseline được tính mới từ empty trong bộ dữ liệu này. Khi kiểm tra một file empty, baseline loại chính file đó. Đây là khảo sát, không phải train/test.</p>
<p>Tham số quạt, độ cao board và hướng đi trong bố trí mới chưa được xác nhận. Các hình biểu diễn biên độ CSI, không phải pha góc của số phức.</p>
<p><a href="quality_and_verification.csv">Chất lượng từng recording</a> | <a href="session_medians.csv">Đặc trưng từng recording</a> | <a href="all_window_features.csv">Toàn bộ cửa sổ</a> | <a href="verification.json">Kiểm tra công thức độc lập</a></p>
<table><tr><th>ID</th><th>File / mở kết quả</th><th>Gói</th><th>Giây</th><th>Hz</th><th>Gap lớn nhất (ms)</th><th>Thiếu seq RX</th><th>Công thức</th></tr>'''+''.join(blocks)+'</table>'
    page+=''.join(f'<h2>{title}</h2><a href="{fn}"><img src="{fn}"></a>' for fn,title in [
        ('recording_feature_comparison.png','So sánh đặc trưng theo recording'),
        ('all_smoothed_heatmaps.png','Heatmap sau chuẩn hóa và làm mượt'),
        ('all_baseline_heatmaps.png','Sai lệch so với empty baseline')])
    page+='<p>Bản xuất tham khảo làm mượt toàn đoạn. Pipeline train hiện tại làm mượt từng cửa sổ; hai cách có khác biệt ở biên cửa sổ. Không đưa trực tiếp bảng xuất này vào model cũ.</p>'
    (root/'index.html').write_text(page,encoding='utf-8')
    # Compact per-file interpretation: formulas can be checked, labels cannot.
    perfile=med[['id','file','label','temporal_std_mean','temporal_absdiff_mean',
                 'baseline_rmse','baseline_corr','motion_bandpower_mean']].merge(
                     audit[['file','duration_esp_s','rate_hz','feature_formulas_pass']],on='file')
    perfile.to_csv(root/'per_file_assessment.csv',index=False)
    with zipfile.ZipFile(root.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root))
    print(audit[['id','label','packets','duration_esp_s','rate_hz','seq_missing','rssi_mean']].to_string(index=False),flush=True)
    print(med[['id','label',*selected]].to_string(index=False),flush=True)
    print('ALL CHECKS PASSED',len(checks),'recordings',len(all_features),'windows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root)
