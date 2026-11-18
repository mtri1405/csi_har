"""Inspect raw CSV quality without changing or relabeling recordings."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from csi.signal import load_receiver_csv
from csi.paths import DATA_ROOT, REPORT_ROOT

def audit(roots):
    results=[]
    for root in roots:
        for path in sorted(Path(root).rglob('*.csv')):
            item={'path':str(path.resolve()),'folder_label':path.parent.name,
                  'filename_label':path.stem.split('_')[0]}
            try:
                frame=pd.read_csv(path)
                item['rows']=len(frame)
                item['length_counts']=frame['len'].value_counts().to_dict()
                item['label_conflict']=item['folder_label'] != item['filename_label']
                data=load_receiver_csv(path)
                item.update(valid_rows=len(data.time_s),duration_s=float(data.time_s[-1]),
                    rate_hz=float((len(data.time_s)-1)/data.time_s[-1]),
                    max_gap_s=float(np.diff(data.time_s).max()))
            except Exception as exc:
                item['error']=str(exc)
            results.append(item)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',type=Path,default=[DATA_ROOT])
    p.add_argument('--out',type=Path,default=REPORT_ROOT/'data_audit.json')
    args=p.parse_args()
    rows=audit(args.roots)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(rows,indent=2,ensure_ascii=False),flush=True)
