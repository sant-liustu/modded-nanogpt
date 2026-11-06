"""Pair first, then average pulse-minus-baseline validation loss on the exact common tau grid."""
import argparse,csv,json,math,statistics
from pathlib import Path

def aggregate(root):
    root=Path(root);plan=json.loads((root/'manifest.json').read_text());seeds=plan['replicates']
    if len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('need distinct replicate IDs')
    if len(plan['jobs'])!=2*len(seeds):raise ValueError('manifest must contain exactly one pair per seed')
    if plan['fork_step']!=16000:raise ValueError('wrong fork point')
    curves={};fingerprints={};starts={}
    for job in plan['jobs']:
        seed=job['replicate_id'];arm=job['experiment'];key=(seed,arm)
        if seed not in seeds or arm not in ('baseline','elr001') or key in curves:raise ValueError('duplicate/unknown job')
        # Relative to run root so downloaded/moved results remain aggregatable.
        log_root=root/'logs'/job['name']
        paths=sorted(log_root.rglob('probe_metrics.jsonl'))
        metas=sorted(log_root.rglob('mix_data_metadata.json'))
        if not paths or not metas:raise ValueError('missing logs/metadata: '+job['name'])
        rows={}
        for meta in metas:
            m=json.loads(meta.read_text())
            if m['replicate_id']!=seed or m['fork_step']!=16000:raise ValueError('metadata replicate/fork mismatch')
            value=(tuple(m['mix_shards']),m['context_order_sha256'],m['context_length'])
            if list(value[0])!=job['mix_shards']:raise ValueError('two-shard pool mismatch')
            if key in fingerprints and fingerprints[key]!=value:raise ValueError('shard changed across resume logs')
            fingerprints[key]=value
        for path in paths:
            for line in path.read_text().splitlines():
                row=json.loads(line);i=row['branch_step']
                if row['replicate_id']!=seed or row['experiment']!=arm or row['fork_step']!=16000 or row['val_tokens']!=plan['validation_tokens'] or row['mix_shards']!=job['mix_shards'] or row['step']!=16000+i:raise ValueError('metric identity mismatch')
                if not math.isfinite(row['val_loss']):raise ValueError('nonfinite loss')
                expected_tau=.06*i if arm=='baseline' else .01*min(i,120)+.06*max(i-120,0)
                if not math.isclose(row['intrinsic_time'],expected_tau,abs_tol=1e-9):raise ValueError('intrinsic time mismatch')
                if i in rows and not math.isclose(rows[i]['val_loss'],row['val_loss'],rel_tol=0,abs_tol=1e-7):raise ValueError('conflicting duplicate/resume point')
                rows[i]=row
        last=270 if arm=='baseline' else 370
        if set(rows)!=set(range(last+1)):raise ValueError('incomplete branch: '+job['name'])
        curves[key]=rows;starts[key]=rows[0]['val_loss']
    if len({fingerprints[(seed,'baseline')] for seed in seeds})!=len(seeds):raise ValueError('duplicate mixed context streams across repeats')
    per_pair=[];summary=[]
    for seed in seeds:
        if fingerprints[(seed,'baseline')]!=fingerprints[(seed,'elr001')]:raise ValueError('paired arms do not share the same pool and context order')
        if not math.isclose(starts[(seed,'baseline')],starts[(seed,'elr001')],rel_tol=0,abs_tol=1e-6):raise ValueError('paired initial validation losses differ')
    initial=list(starts.values())
    if max(initial)-min(initial)>1e-6:raise ValueError('repeats do not start at the same measured checkpoint')
    for i in range(271):
        pulse_i=6*i if i<=20 else 120+(i-20)
        deltas=[];baselines=[];pulses=[]
        for seed in seeds:
            b=curves[(seed,'baseline')][i]['val_loss'];p=curves[(seed,'elr001')][pulse_i]['val_loss'];delta=p-b
            deltas.append(delta);baselines.append(b);pulses.append(p)
            per_pair.append(dict(replicate_id=seed,pool_shards=str(fingerprints[(seed,'baseline')][0]),intrinsic_time=.06*i,baseline_step=i,pulse_step=pulse_i,baseline_val_loss=b,pulse_val_loss=p,delta_loss=delta))
        mean=statistics.mean(deltas);sd=statistics.stdev(deltas);se=sd/math.sqrt(len(seeds))
        summary.append(dict(intrinsic_time=.06*i,recovery_intrinsic_time=.06*i-1.2,n=len(seeds),mean_baseline_val_loss=statistics.mean(baselines),mean_pulse_val_loss=statistics.mean(pulses),mean_delta_loss=mean,sd_delta_loss=sd,se_delta_loss=se,normal95_lower=mean-1.96*se,normal95_upper=mean+1.96*se))
    out=root/'aggregate';out.mkdir(exist_ok=True)
    for name,rows in [('per_pair_delta.csv',per_pair),('mean_delta.csv',summary)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(pairs=len(seeds),grid_points=271,delta_sign='pulse minus baseline',alignment='exact intrinsic time: baseline i, pulse6i throughi20 theni+100',uncertainty='sample SD and SE across paired repeats; normal95 is descriptive pointwise, not simultaneous',validation_tokens=plan['validation_tokens'],conditional_on='one shared stable16000 checkpoint; mixed-two-shard shuffled-context trajectories;separate protocol from old sequential repeats'),indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
    rows=aggregate(a.run_root);print(f"Aggregated {rows[0]['n']} pairs on {len(rows)} points: {a.run_root/'aggregate'/'mean_delta.csv'}")
if __name__=='__main__':main()
