"""Stdlib verification for400 pairs per small model and exact paired statistics."""
import ast,hashlib,json,math,random,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from run_small_mix2 import make_plan
models=json.loads((root/'models.json').read_text());reference=json.loads((root/'provenance.json').read_text())['inherited_source_segments'];reports={}
for model,spec in models.items():
    text=(root/spec['script']).read_text(encoding='utf-8');tree=ast.parse(text)
    for name,segment in reference.items():
        node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.dump(node)==ast.dump(ast.parse(segment).body[0]),(model,name)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Hyperparameters');cfg={n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign)}
    assert (cfg['model_width'],cfg['model_depth'],cfg['expected_world_size'],cfg['device_batch_size'])==(spec['width'],spec['depth'],1,128)
    assert spec['parent_runtime_sha256'] in text
    prepare=(root/spec['prepare_script']).read_text(encoding='utf-8');assert hashlib.sha256(prepare.encode()).hexdigest()==spec['parent_runtime_sha256']
    ns=dict(random=random,args=SimpleNamespace(warmup_iters=924,peak_elr=.06))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('make_context_order','shared_elr')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual-functions>','exec'),ns)
    hashes=[]
    for seed in range(4000,4400):
        before=random.getstate();order=ns['make_context_order']([1001,1201],seed)
        assert before==random.getstate() and sorted(order)==list(range(2202)) and order==ns['make_context_order']([1001,1201],seed)
        hashes.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())
    assert len(set(hashes))==400
    setup=tree.body[next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='cli' for v in n.targets))+1:]
    assigns=[n for n in setup if isinstance(n,ast.Assign) and any(isinstance(v,ast.Attribute) and v.attr in ('stage_start','suite_mode','pulse_elr','pulse_updates','stop_step') for v in n.targets)]
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    save=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='save_checkpoint')
    stop=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=='step == cli.stop_step')
    for arm,width,total in [('baseline',0,270),('elr001',120,370)]:
        cli=SimpleNamespace(fork_step=16000,experiment=arm,stop_after_save=False,save_at=None);ns['cli']=cli
        exec(compile(ast.Module(body=assigns,type_ignores=[]),'<actual-setup>','exec'),ns)
        values=[ns['shared_elr'](u) for u in range(16001,cli.stop_step+1)];assert values==[.01]*width+[.06]*(total-width) and math.isclose(sum(values),16.2)
        for saving in (False,True):
            updates=[];saved=[];ns.update(x=None,y=None,updates=updates,checkpoints=({16120,16370} if width else {16270}),save_fork_checkpoint=lambda step,x,y:saved.append(step))
            body=ast.parse('save_checkpoint=step in checkpoints' if saving else 'save_checkpoint=False').body+[save,stop]+ast.parse('updates.append(step+1)').body
            test=ast.For(target=ast.Name(id='step',ctx=ast.Store()),iter=ast.parse(f'range(16000,{cli.stop_step+2})',mode='eval').body,body=body,orelse=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-stop>','exec'),ns)
            assert updates==list(range(16001,cli.stop_step+1)) and saved==(sorted(ns['checkpoints']) if saving else [])
    info=dict(ranks=1,device_batch_size=128,context=1024,batch_size=128,compile_model=1,input_bin='train*.bin',input_val_bin='val*.bin',train_shard=20,train_files=[str(i) for i in range(100)],train_sizes=[1024+2*100000000]*100)
    plan=make_plan(info,'/mock/shared16000','/mock/output',model=model)
    assert len(plan['jobs'])==800 and len(plan['replicates'])==400 and plan['model']==model
    for seed in plan['replicates']:
        pair=[j for j in plan['jobs'] if j['replicate_id']==seed]
        assert len(pair)==2 and pair[0]['mix_shards']==pair[1]['mix_shards'] and min(pair[0]['mix_shards'])>20
        assert all(spec['script'] in ' '.join(j['command']) for j in pair)
    fixed=make_plan(info,'/mock/shared','/mock/fixed',model=model,fixed_shards=[21,22]);assert all(j['mix_shards']==[21,22] for j in fixed['jobs'])
    pns=dict(args=SimpleNamespace(warmup_iters=924,peak_elr=.06),cli=SimpleNamespace(stage_start=15000,experiment='stable',pulse_updates=0))
    pfun=next(n for n in ast.parse(prepare).body if isinstance(n,ast.FunctionDef) and n.name=='shared_elr')
    exec(compile(ast.Module(body=[pfun],type_ignores=[]),'<prepare-elr>','exec'),pns)
    assert all(pns['shared_elr'](u)==.06 for u in range(15001,16001))
    doc=dict(version=1,steps=[dict(update_step=u,default_elr=pns['shared_elr'](u)) for u in range(1,18859)])
    assert hashlib.sha256(json.dumps(doc,indent=2).encode()).hexdigest()=='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'
    reports[model]=dict(runs=len(plan['jobs']),pairs=400,context_order_fingerprints=hashes)
assert sum(v['runs'] for v in reports.values())==1600
for p in root.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
# Known mean/SD/SE,model labels and exact271-point grid,run independently per model.
from aggregate_mix_pairs import aggregate
with tempfile.TemporaryDirectory(dir=root) as tmp:
    for model in models:
        out=Path(tmp)/model;out.mkdir();plan=dict(model=model,fork_step=16000,replicates=[1,2,3],validation_tokens=1048576,jobs=[])
        for seed in (1,2,3):
            for arm in ('baseline','elr001'):
                name=f'seed{seed}_{arm}';log=out/'logs'/name/'uuid';log.mkdir(parents=True)
                plan['jobs'].append(dict(name=name,replicate_id=seed,experiment=arm,mix_shards=[21,22]))
                (log/'mix_data_metadata.json').write_text(json.dumps(dict(model_key=model,replicate_id=seed,fork_step=16000,mix_shards=[21,22],context_order_sha256=str(seed),context_length=1024)))
                rows=[]
                for i in range((270 if arm=='baseline' else 370)+1):
                    tau=.06*i if arm=='baseline' else .01*min(i,120)+.06*max(i-120,0)
                    rows.append(dict(model_key=model,replicate_id=seed,experiment=arm,fork_step=16000,mix_shards=[21,22],val_tokens=1048576,step=16000+i,branch_step=i,intrinsic_time=tau,val_loss=3-.001*tau+(seed*.001*tau if arm=='elr001' else 0)))
                (log/'probe_metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        (out/'manifest.json').write_text(json.dumps(plan));stats=aggregate(out)
        assert len(stats)==271
        for r in stats:
            t=r['intrinsic_time'];assert r['model']==model and math.isclose(r['mean_delta_loss'],.002*t,abs_tol=1e-14) and math.isclose(r['sd_delta_loss'],.001*t,abs_tol=1e-14) and math.isclose(r['se_delta_loss'],.001*t/math.sqrt(3),abs_tol=1e-14)
(root/'verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,models=reports,total_runs=1600,scope='per-model defaults/parent binding,400 private permutations,800 paired commands each,actual250/270/370 schedules and both save-policy stops,unchanged prep source and stable16k schedule,known per-model stats'),indent=2),encoding='utf-8')
print('PASS:400 pairs/model,1600 planned runs,model/source binding,stops and paired stats')
