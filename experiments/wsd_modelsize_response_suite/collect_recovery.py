"""Collect historical-probe recovery deltas on the exact shared intrinsic-time grid."""
import argparse,csv,json,math
from pathlib import Path
WIDTHS={'elr005':24,'elr004':30,'elr003':40,'elr002':60,'elr001':120}
ELRS={'elr005':.05,'elr004':.04,'elr003':.03,'elr002':.02,'elr001':.01}

def collect(root):
    root=Path(root);plan=json.loads((root/'manifest.json').read_text());curves={}
    for job in plan['jobs']:
        if job['suite_mode']!='historical' or job['experiment']=='stable':continue
        key=(job['model'],job['fork_step'],job['experiment']);rows={}
        for path in sorted((root/job['model']/'probes'/'logs'/job['name']).rglob('probe_metrics.jsonl')):
            for line in path.read_text().splitlines():
                row=json.loads(line);step=row['branch_step']
                if row['model_key']!=key[0] or row['fork_step']!=key[1] or row['experiment']!=key[2] or row['suite_mode']!='historical' or not math.isfinite(row['val_loss']):raise ValueError('historical metric identity/nonfinite mismatch')
                if step in rows and not math.isclose(rows[step]['val_loss'],row['val_loss'],rel_tol=0,abs_tol=1e-7):raise ValueError('conflicting resume point')
                rows[step]=row
        last=520 if key[2]=='baseline' else WIDTHS[key[2]]+500
        if set(rows)!=set(range(last+1)):raise ValueError('incomplete historical probe: '+str(key))
        curves[key]=rows
    output=[]
    for (model,fork,arm),pulse in curves.items():
        if arm=='baseline':continue
        baseline=curves[(model,fork,'baseline')];width=WIDTHS[arm];eta=ELRS[arm]
        if pulse[0]['val_tokens']!=baseline[0]['val_tokens'] or not math.isclose(pulse[0]['val_loss'],baseline[0]['val_loss'],rel_tol=0,abs_tol=1e-6):raise ValueError('historical branches do not share initial measured state/protocol')
        for k in range(501):
            b=baseline[20+k];p=pulse[width+k]
            if not math.isclose(b['intrinsic_time'],p['intrinsic_time'],abs_tol=1e-9):raise ValueError('intrinsic time mismatch')
            delta=p['val_loss']-b['val_loss']
            output.append(dict(model=model,fork_step=fork,experiment=arm,pulse_elr=eta,pulse_updates=width,recovery_step=k,recovery_intrinsic_time=.06*k,baseline_val_loss=b['val_loss'],pulse_val_loss=p['val_loss'],delta_loss=delta,delta_per_elr_change=delta/(eta-.06)))
    if output:
        with (root/'historical_recovery_delta.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args();rows=collect(a.run_root);print(f'Collected {len(rows)} recovery-delta rows')
if __name__=='__main__':main()
