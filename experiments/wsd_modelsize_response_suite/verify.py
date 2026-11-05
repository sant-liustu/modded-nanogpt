"""Dependency-free checks for all five model scripts,scope counts and checkpoint dependencies."""
import ast,hashlib,importlib.util,json,math,random,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from run_suite import make_plan
models=json.loads((root/'models.json').read_text());provenance=json.loads((root/'provenance.json').read_text())
reports=[]
for key,spec in models.items():
    path=root/spec['script'];text=path.read_text(encoding='utf-8');tree=ast.parse(text)
    for name,segment in provenance['inherited_source_segments'].items():
        n=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.dump(n)==ast.dump(ast.parse(segment).body[0]),(key,name)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Hyperparameters')
    cfg={n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign)}
    assert (cfg['model_width'],cfg['model_depth'],cfg['expected_world_size'],cfg['device_batch_size'])==(spec['width'],spec['depth'],spec['default_ranks'],spec['default_device_batch'])
    assert cfg['expected_world_size']*cfg['device_batch_size']==128 and cfg['peak_elr']==.06 and cfg['gamma_learning_rate']==.0018
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('shared_elr','select_fresh_shard')]
    setup=tree.body[next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='cli' for v in n.targets))+1:]
    assignments=[n for n in setup if isinstance(n,ast.Assign) and any(isinstance(v,ast.Attribute) and v.attr in ('stage_start','suite_mode','pulse_elr','pulse_updates','stop_step') for v in n.targets)]
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step');save=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='save_checkpoint')
    cases=[(15000,'stable',0,1000)]+[(step,arm,width,520 if not width else width+500) for step in (5000,10000,15000,16000) for arm,width in [('baseline',0),('elr005',24),('elr004',30),('elr003',40),('elr002',60),('elr001',120)]]
    for step,arm,width,total in cases:
        cli=SimpleNamespace(fork_step=step,experiment=arm,fresh_replicate=-1,shard_seed=0,fresh_repeats=20,stop_after_save=False,save_at=None)
        ns=dict(cli=cli,args=SimpleNamespace(**cfg),random=random,hashlib=hashlib,json=json)
        exec(compile(ast.Module(body=assignments+functions,type_ignores=[]),'<actual-setup>','exec'),ns)
        assert cli.stage_start==step and cli.stop_step==step+total and cli.pulse_updates==width
        values=[ns['shared_elr'](u) for u in range(step+1,cli.stop_step+1)]
        assert values==[cli.pulse_elr]*width+[.06]*(total-width)
        assert math.isclose(sum(values),60 if arm=='stable' else 31.2,rel_tol=1e-12)
        if width:assert math.isclose(width*cli.pulse_elr,1.2)
        saved=[];updates=[];ns.update(x=None,y=None,checkpoints=({step+width,cli.stop_step} if width else {cli.stop_step}),updates=updates,save_fork_checkpoint=lambda step,x,y:saved.append(step))
        test=ast.For(target=ast.Name(id='step',ctx=ast.Store()),iter=ast.parse(f'range({step},{cli.stop_step+2})',mode='eval').body,body=ast.parse('save_checkpoint=step in checkpoints').body+[save]+ast.parse('updates.append(step+1)').body,orelse=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-stop>','exec'),ns)
        assert updates==list(range(step+1,cli.stop_step+1)) and saved==sorted(ns['checkpoints'])
        if arm=='stable':
            doc=dict(version=1,steps=[dict(update_step=u,default_elr=ns['shared_elr'](u)) for u in range(1,18859)])
            assert hashlib.sha256(json.dumps(doc,indent=2).encode()).hexdigest()=='9e3a4b48e9c2de544e2c1c7f35775c80f910b39717eed85ac45645434961c387'
    cli.fresh_replicate=0;cli.suite_mode='fresh';cli.stage_start=16000
    fake=dict(config=dict(batch_size=128),train_loader=dict(B=64,T=1024,world=2,shard=20,sizes=[1024+2*100000000]*100,files=[str(i) for i in range(100)]))
    selection=[];before=random.getstate()
    for rep in range(20):cli.fresh_replicate=rep;info,excluded=ns['select_fresh_shard'](fake);selection.append(info['index']);assert excluded==list(range(21))
    assert len(set(selection))==20 and all(i>20 for i in selection) and before==random.getstate()
    assert 'val_loader.reset()' in text and 'if True:  # Fixed-prefix validation' in text
    reports.append(dict(model=key,cases=len(cases),default_ranks=spec['default_ranks'],status='PASS'))
for p in root.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp)
    config=dict(models={key:dict(checkpoints={str(step):f'/mock/{key}/{step}' for step in (5000,10000,15000,16000)}) for key in models})
    def info(path,key,step):
        m=models[key];return dict(ranks=m['default_ranks'],device_batch_size=m['default_device_batch'],compile_model=1,input_bin='train*.bin',input_val_bin='val*.bin')
    plan=make_plan(config,tmp/'all',['time','amplitude','repeats'],list(models),info)
    assert len(plan['jobs'])==275
    for key in models:
        assert plan['models'][key]==dict(prepare=0,historical=15,fresh=40,total=55)
        historical=[j for j in plan['jobs'] if j['model']==key and j['suite_mode']=='historical']
        assert len({(j['fork_step'],j['experiment']) for j in historical})==15
        for j in historical:
            if j['fork_step']!=16000:assert j['experiment'] in ('baseline','elr001','elr003')
        assert len(plan['repeat_manifests'][key]['jobs'])==40
    for entry in config['models'].values():entry['checkpoints'].pop('16000')
    prepared=make_plan(config,tmp/'prepare',['time','amplitude','repeats'],list(models),info)
    assert len(prepared['jobs'])==280
    for key in models:
        jobs=[j for j in prepared['jobs'] if j['model']==key];assert jobs[0]['experiment']=='stable'
        expected=str((tmp/'prepare'/key/'checkpoints'/'prepare16000'/'step016000').resolve())
        assert all(j['input_checkpoint']==expected for j in jobs if j['fork_step']==16000)
    time_only=make_plan(config,tmp/'time',['time'],['w768_d12'],info);assert len(time_only['jobs'])==13
    amp=make_plan(config,tmp/'amp',['amplitude'],['w768_d12'],info);assert len(amp['jobs'])==7
    # New seed1 batch excludes all20 shards selected by seed0,without trainer changes.
    def extra_info(path,key,step):
        return dict(info(path,key,step),train_shard=20,train_files=[str(i) for i in range(100)])
    extra_config=dict(models={'w768_d12':dict(checkpoints={'16000':'/mock/shared16000'})})
    extra=make_plan(extra_config,tmp/'extra',['repeats'],['w768_d12'],extra_info,20,1,1048576,[0])
    manifest=extra['repeat_manifests']['w768_d12'];jobs=manifest['jobs']
    assert len(jobs)==40 and len(manifest['replicates'])==20
    selected={j['fresh_shard_index'] for j in jobs}
    assert len(selected)==20 and not(selected & set(manifest['excluded_previous_shard_indices']))
    assert len(manifest['excluded_previous_shard_indices'])==20
    from run_suite import additional_positions
    try:additional_positions(dict(train_shard=0,train_files=list(range(30))),1,[0],20)
    except ValueError:pass
    else:raise AssertionError('reused old shards when too few remain')
    for job in jobs:
        c=job['command'];rep=int(c[c.index('--fresh-replicate')+1]);total=int(c[c.index('--fresh-repeats')+1])
        assert 0<=rep<total and total==max(manifest['replicates'])+1
        cli.fresh_replicate=rep;cli.fresh_repeats=total;cli.shard_seed=1
        expected,_=ns['select_fresh_shard'](fake)
        assert expected['index']==job['fresh_shard_index']
    # Exact recovery alignment for every amplitude, with known analytic delta.
    from collect_recovery import collect
    run=tmp/'recovery';run.mkdir();probe_jobs=[]
    for arm,width in [('baseline',0),('elr005',24),('elr004',30),('elr003',40),('elr002',60),('elr001',120)]:
        name=arm;log=run/'w768_d12'/'probes'/'logs'/name/'uuid';log.mkdir(parents=True)
        probe_jobs.append(dict(model='w768_d12',fork_step=16000,experiment=arm,suite_mode='historical',name=name))
        rows=[]
        eta=1.2/width if width else .06
        for i in range((width+500 if width else 520)+1):
            tau=eta*min(i,width)+.06*max(i-width,0)
            # Shared initial loss, then a prescribed recovery response.
            delta=(eta-.06)*.001*tau if width else 0
            rows.append(dict(model_key='w768_d12',fork_step=16000,experiment=arm,suite_mode='historical',branch_step=i,intrinsic_time=tau,val_tokens=1048576,val_loss=3-.001*tau+delta))
        (log/'probe_metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    (run/'manifest.json').write_text(json.dumps(dict(jobs=probe_jobs)))
    rows=collect(run);assert len(rows)==5*501
    for row in rows:assert math.isclose(row['delta_per_elr_change'],.001*(1.2+row['recovery_intrinsic_time']),abs_tol=1e-12)
(root/'verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,model_cases=reports,full_suite_jobs=275,missing_all16000_jobs=280,scope='all model configurations,125 actual schedule/stop cases,shared16k hash,20 shard selection,scope/dedup/dependencies and inherited definitions'),indent=2),encoding='utf-8')
print('PASS:five models,125 schedule/stop cases,275-job scope and16000 preparation')

