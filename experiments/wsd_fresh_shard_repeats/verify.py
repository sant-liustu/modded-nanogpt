"""Stdlib tests of history proof, untouched-shard plan, inherited training code and stats."""
import ast,importlib.util,json,math,os,struct,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from fresh_shards import audit_history,check_fresh_shard,KNOWN_SHARED_SCHEDULE
from run_repeats import make_plan
source=root/'train_fresh_repeat.py';text=source.read_text(encoding='utf-8');tree=ast.parse(text)
provenance=json.loads((root/'provenance.json').read_text())
for name,segment in provenance['inherited_source_segments'].items():
    actual=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    assert ast.dump(actual)==ast.dump(ast.parse(segment).body[0]),name
for p in root.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp);files=[]
    for i,count in enumerate([270001]+[10001]*20):
        p=tmp/f'train{i:03d}.bin';files.append(str(p.resolve()))
        with p.open('wb') as f:f.write(struct.pack('<3i',20240520,1,count)+bytes(1012));f.truncate(1024+2*count)
    saved=dict(files=files,sizes=[os.path.getsize(f) for f in files],modified_ns=[os.stat(f).st_mtime_ns for f in files],B=2,T=8,world=1,rank=0,shard=0,position=16001*16)
    state=dict(source_sha256='bf1ba81bad94b4dad48b10d441730286d10f7f1bf456f02f5db5ee8c32930e37',schedule_sha256=KNOWN_SHARED_SCHEDULE,step=16000,probe_experiment='stable',config=dict(batch_size=2,sequence_length=8),train_loader=saved)
    audit=audit_history(state);assert audit['excluded_indices']==[0] and audit['unvisited_indices']==list(range(1,21))
    args=SimpleNamespace(nproc_per_node=1,device_batch_size=2,probe_val_tokens=32,repeats=20,fresh_shard_indices=None,output=tmp/'run',resume=tmp/'step016000',input_bin=str(tmp/'train*.bin'),input_val_bin='val*.bin')
    plan=make_plan(state,args);assert len(plan['jobs'])==40 and sorted(plan['selected_shard_indices'])==list(range(1,21))
    args.shard_seed=0
    assert make_plan(state,args)['selected_shard_indices']==plan['selected_shard_indices']
    args.shard_seed=1
    assert make_plan(state,args)['selected_shard_indices']!=plan['selected_shard_indices']
    for rep in range(20):
        pair=[j for j in plan['jobs'] if j['replicate_id']==rep];assert len(pair)==2 and pair[0]['fresh_shard_index']==pair[1]['fresh_shard_index']
    try:check_fresh_shard(state,0,620)
    except ValueError:pass
    else:raise AssertionError('partially used shard accepted')
    args.repeats=21
    try:make_plan(state,args)
    except ValueError:pass
    else:raise AssertionError('reused shard to fill missing repeat')
    # Real schedule setup and stop block.
    after=tree.body[next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='cli' for v in n.targets))+1:]
    # The first if validates CLI ids; execute setup through stage/stop assignments explicitly.
    assignments=[n for n in after if isinstance(n,ast.Assign) and any(isinstance(v,ast.Attribute) and v.attr in ('stage_start','pulse_elr','pulse_updates','stop_step') for v in n.targets)]
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shared_elr')
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step');save=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='save_checkpoint')
    for arm,width,total in [('baseline',0,520),('elr001',120,620)]:
        cli=SimpleNamespace(fork_step=16000,experiment=arm,stop_after_save=False,save_at=None)
        ns=dict(cli=cli,args=SimpleNamespace(warmup_iters=924,peak_elr=.06))
        exec(compile(ast.Module(body=assignments+[fun],type_ignores=[]),'<actual-schedule>','exec'),ns)
        values=[ns['shared_elr'](i) for i in range(16001,cli.stop_step+1)]
        assert values==[.01]*width+[.06]*(total-width) and math.isclose(sum(values),31.2)
        saved_steps=[];updates=[];ns.update(x=None,y=None,checkpoints=({16120,16620} if width else {16520}),save_fork_checkpoint=lambda step,x,y:saved_steps.append(step),updates=updates)
        test=ast.For(target=ast.Name(id='step',ctx=ast.Store()),iter=ast.parse(f'range(16000,{cli.stop_step+2})',mode='eval').body,body=ast.parse('save_checkpoint=step in checkpoints').body+[save]+ast.parse('updates.append(step+1)').body,orelse=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-stop>','exec'),ns)
        assert updates==list(range(16001,cli.stop_step+1)) and saved_steps==sorted(ns['checkpoints'])
    # Synthetic three-shard known response; no interpolation.
    run=tmp/'stats';run.mkdir();synthetic=dict(fork_step=16000,replicates=[0,1,2],validation_tokens=1048576,jobs=[])
    for rep in synthetic['replicates']:
        for arm in ('baseline','elr001'):
            name=f'rep{rep}_{arm}';log=run/'logs'/name/'uuid';log.mkdir(parents=True)
            synthetic['jobs'].append(dict(name=name,replicate_id=rep,fresh_shard_index=rep+1,experiment=arm))
            (log/'fresh_shard_metadata.json').write_text(json.dumps(dict(replicate_id=rep,fork_step=16000,fresh_shard=dict(index=rep+1,path=str(rep+1),tokens=100000000),excluded_shard_indices=[0])))
            rows=[]
            for i in range((520 if arm=='baseline' else 620)+1):
                tau=.06*i if arm=='baseline' else .01*min(i,120)+.06*max(i-120,0)
                rows.append(dict(replicate_id=rep,fresh_shard_index=rep+1,experiment=arm,fork_step=16000,val_tokens=1048576,step=16000+i,branch_step=i,intrinsic_time=tau,val_loss=3-.001*tau+((rep+1)*.001*tau if arm=='elr001' else 0)))
            (log/'probe_metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (run/'manifest.json').write_text(json.dumps(synthetic))
    from aggregate_pairs import aggregate
    rows=aggregate(run)
    for row in rows:
        tau=row['intrinsic_time'];assert math.isclose(row['mean_delta_loss'],.002*tau,abs_tol=1e-14) and math.isclose(row['sd_delta_loss'],.001*tau,abs_tol=1e-14) and math.isclose(row['se_delta_loss'],.001*tau/math.sqrt(3),abs_tol=1e-14)
assert 'random.Random' not in text and 'val_loader.reset()' in text
(root/'verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,scope='inherited definitions;metadata-only suffix;current/previous shard exclusion;reproducible shard-only shuffle;20 distinct untouched shards/40 jobs;actual schedules/stops;known paired mean/SD/SE;no shuffling'),indent=2),encoding='utf-8')
print('PASS:remaining-shard shuffle,20 pairs/40 jobs,schedules and paired statistics')
