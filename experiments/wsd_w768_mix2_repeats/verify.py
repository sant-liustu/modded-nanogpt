"""Stdlib tests:140 unique context orders,280-job plan,ELR boundaries and paired stats."""
import ast,hashlib,json,math,random,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from run_mix2 import make_plan
source=root/'train_w768_mix2.py';text=source.read_text(encoding='utf-8');tree=ast.parse(text);provenance=json.loads((root/'provenance.json').read_text())
for name,segment in provenance['inherited_source_segments'].items():
    actual=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    assert ast.dump(actual)==ast.dump(ast.parse(segment).body[0]),name
for p in root.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
ns=dict(random=random)
order_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_context_order')
exec(compile(ast.Module(body=[order_fn],type_ignores=[]),'<actual-context-order>','exec'),ns)
digests=[]
for seed in range(2000,2140):
    before=random.getstate();order=ns['make_context_order']([1001,1201],seed)
    assert before==random.getstate() and sorted(order)==list(range(2202))
    assert order==ns['make_context_order']([1001,1201],seed)
    digests.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())
assert len(set(digests))==140
info=dict(ranks=2,device_batch_size=64,context=1024,batch_size=128,compile_model=1,input_bin='train*.bin',input_val_bin='val*.bin',train_shard=20,train_files=[str(i) for i in range(100)],train_sizes=[1024+2*100000000]*100)
plan=make_plan(info,'/mock/shared16000','/mock/output')
assert len(plan['jobs'])==280 and len(plan['replicates'])==140
for seed in plan['replicates']:
    pair=[j for j in plan['jobs'] if j['replicate_id']==seed]
    assert len(pair)==2 and pair[0]['mix_shards']==pair[1]['mix_shards'] and len(set(pair[0]['mix_shards']))==2 and min(pair[0]['mix_shards'])>20
fixed=make_plan(info,'/mock/shared16000','/mock/fixed',fixed_shards=[21,22]);assert all(j['mix_shards']==[21,22] for j in fixed['jobs'])
try:make_plan(info,'/mock/shared','/mock/bad',fixed_shards=[20,21])
except ValueError:pass
else:raise AssertionError('used shard accepted')
setup=tree.body[next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='cli' for v in n.targets))+1:]
assignments=[n for n in setup if isinstance(n,ast.Assign) and any(isinstance(v,ast.Attribute) and v.attr in ('stage_start','suite_mode','pulse_elr','pulse_updates','stop_step') for v in n.targets)]
fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shared_elr')
loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step');save=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='save_checkpoint')
for arm,width,total in [('baseline',0,270),('elr001',120,370)]:
    cli=SimpleNamespace(fork_step=16000,experiment=arm,stop_after_save=False,save_at=None)
    ctx=dict(cli=cli,args=SimpleNamespace(warmup_iters=924,peak_elr=.06))
    exec(compile(ast.Module(body=assignments+[fun],type_ignores=[]),'<actual-schedule>','exec'),ctx)
    values=[ctx['shared_elr'](u) for u in range(16001,cli.stop_step+1)]
    assert values==[.01]*width+[.06]*(total-width) and math.isclose(sum(values),16.2)
    updates=[];saved=[];ctx.update(x=None,y=None,checkpoints=({16120,16370} if width else {16270}),updates=updates,save_fork_checkpoint=lambda step,x,y:saved.append(step))
    test=ast.For(target=ast.Name(id='step',ctx=ast.Store()),iter=ast.parse(f'range(16000,{cli.stop_step+2})',mode='eval').body,body=ast.parse('save_checkpoint=step in checkpoints').body+[save]+[next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=='step == cli.stop_step')]+ast.parse('updates.append(step+1)').body,orelse=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-stop>','exec'),ctx)
    assert updates==list(range(16001,cli.stop_step+1)) and saved==sorted(ctx['checkpoints'])
    updates.clear();saved.clear()
    test.body[0]=ast.parse('save_checkpoint=False').body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-metrics-only-stop>','exec'),ctx)
    assert updates==list(range(16001,cli.stop_step+1)) and saved==[]
with tempfile.TemporaryDirectory(dir=root) as tmp:
    out=Path(tmp);synthetic=dict(fork_step=16000,replicates=[1,2,3],validation_tokens=1048576,jobs=[])
    for seed in (1,2,3):
        for arm in ('baseline','elr001'):
            name=f'seed{seed}_{arm}';log=out/'logs'/name/'uuid';log.mkdir(parents=True)
            synthetic['jobs'].append(dict(name=name,replicate_id=seed,experiment=arm,mix_shards=[21,22]))
            (log/'mix_data_metadata.json').write_text(json.dumps(dict(replicate_id=seed,fork_step=16000,mix_shards=[21,22],context_order_sha256=str(seed),context_length=1024)))
            rows=[]
            for i in range((270 if arm=='baseline' else 370)+1):
                tau=.06*i if arm=='baseline' else .01*min(i,120)+.06*max(i-120,0)
                rows.append(dict(replicate_id=seed,experiment=arm,fork_step=16000,mix_shards=[21,22],val_tokens=1048576,step=16000+i,branch_step=i,intrinsic_time=tau,val_loss=3-.001*tau+(seed*.001*tau if arm=='elr001' else 0)))
            (log/'probe_metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (out/'manifest.json').write_text(json.dumps(synthetic))
    from aggregate_mix_pairs import aggregate
    stats=aggregate(out)
    for r in stats:
        t=r['intrinsic_time'];assert math.isclose(r['mean_delta_loss'],.002*t,abs_tol=1e-14) and math.isclose(r['sd_delta_loss'],.001*t,abs_tol=1e-14) and math.isclose(r['se_delta_loss'],.001*t/math.sqrt(3),abs_tol=1e-14)
(root/'verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,recovery_updates=250,baseline_updates=270,pulse_updates=370,total_intrinsic_time=16.2,context_order_fingerprints=digests,scope='inherited model/optimizer/ELR;140 deterministic unique context permutations;280 paired commands;actual schedules/stops;known mixed-protocol delta mean/SD/SE'),indent=2),encoding='utf-8')
print('PASS:140 context permutations,280-run plan,ELR/stops and paired statistics')
