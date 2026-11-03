"""Check actual schedules, unchanged dynamics and exact executed stop boundaries."""
import ast,hashlib,json,math,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
root=Path(__file__).resolve().parent
p=root/'train_w768_d12_continue.py'
tree=ast.parse(p.read_text())
prov=json.loads((root/'provenance.json').read_text())
original=(root/prov['source']).resolve().read_text(encoding='utf-8')
assert hashlib.sha256(original.encode()).hexdigest()==prov['source_sha256']
base=ast.parse(original)
def definitions(t):return {n.name:ast.dump(n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
old,new=definitions(base),definitions(tree)
assert {k:v for k,v in old.items() if k not in ('shared_elr','capture_training_state','restore_training_state')}=={k:v for k,v in new.items() if k not in ('shared_elr','capture_training_state','restore_training_state')}
assert old['Hyperparameters']==new['Hyperparameters']
fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shared_elr')
ns={'args':SimpleNamespace(peak_elr=.06,warmup_iters=924),'cli':SimpleNamespace(experiment='high')}
exec(compile(ast.Module(body=[fun],type_ignores=[]),str(p),'exec'),ns)
basefun=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='shared_elr')
basens={'args':SimpleNamespace(peak_elr=.06,warmup_iters=924,num_iterations=18858,warmdown_iters=3858)}
exec(compile(ast.Module(body=[basefun],type_ignores=[]),'<base>','exec'),basens)
base_schedule=json.dumps(dict(version=1,steps=[dict(update_step=u,default_elr=basens['shared_elr'](u)) for u in range(1,18859)]),indent=2)
assert hashlib.sha256(base_schedule.encode()).hexdigest()==prov['source_schedule_sha256']
loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
save=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='save_checkpoint')
report=[]
for arm,end,integral in [('stable',16000,60),('high',18000,120),('low',18000,12),('pulse',18200,121.2)]:
    ns['cli'].experiment=arm
    for u in range(1,15001):assert ns['shared_elr'](u)==basens['shared_elr'](u)
    start=15000 if arm=='stable' else 16000
    assert all(ns['shared_elr'](u)==.06 for u in range(15001,16001))
    values=[ns['shared_elr'](u) for u in range(start+1,end+1)]
    assert math.isclose(sum(values),integral,rel_tol=1e-12)
    if arm=='pulse': assert values[:200]==[.006]*200 and values[200:]==[.06]*2000
    else:assert values==[.06 if arm in ('stable','high') else .006]*(end-start)
    checkpoints={end}|({16200} if arm=='pulse' else set())
    control={'cli':SimpleNamespace(stop_step=end,stop_after_save=False,save_at=None),'saved':[],'updates':[],'x':None,'y':None,'checkpoints':checkpoints}
    control['save_fork_checkpoint']=lambda step,x,y:control['saved'].append(step)
    body=ast.parse('save_checkpoint = step in checkpoints').body+[save]+ast.parse('updates.append(step+1)').body
    test=ast.For(target=ast.Name(id='step',ctx=ast.Store()),iter=ast.parse(f'range({start},18859)',mode='eval').body,body=body,orelse=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-stop-block>','exec'),control)
    assert control['updates']==list(range(start+1,end+1)) and control['saved']==sorted(checkpoints)
    budget=json.loads(subprocess.check_output([sys.executable,str(p),'--experiment',arm,'--describe-only'],text=True))
    assert budget['stop_step']==end and budget['continuation_updates']==len(values)
    report.append(dict(arm=arm,updates=len(values),intrinsic_time=integral,checkpoints=sorted(checkpoints)))
with tempfile.TemporaryDirectory() as d:
    temp=Path(d);ckpt=temp/'step015000';ckpt.mkdir()
    for r in range(2):(ckpt/f'rank{r:05d}.pt').touch()
    (ckpt/'complete.json').write_text(json.dumps(dict(version=1,step=15000,world_size=2,files=['rank00000.pt','rank00001.pt'])))
    commands=json.loads(subprocess.check_output([sys.executable,str(root/'run_three.py'),'--resume',str(ckpt),'--input-bin','train_*.bin','--input-val-bin','val_*.bin','--output',str(temp/'out'),'--dry-run'],text=True))
    assert [j['experiment'] for j in commands]==['stable','high','low','pulse']
    assert commands[0]['command'][commands[0]['command'].index('--resume')+1]==str(ckpt.resolve())
    expected_shared=str((temp/'out'/'checkpoints'/'stable'/'step016000').resolve())
    assert all(j['command'][j['command'].index('--resume')+1]==expected_shared for j in commands[1:])
    assert all(j['command'][j['command'].index('--eval-every')+1]=='1' for j in commands)
# Evaluation block must be unconditionally executed at each loop boundary, including stage endpoints.
validation=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is True)
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='reset' for n in ast.walk(validation))
for start,end in [(15000,16000),(16000,18000),(16000,18200)]:
    assert sum(bool(eval(compile(ast.Expression(body=validation.test),'<eval-condition>','eval'))) for step in range(start,end+1))==end-start+1
(root/'verification.json').write_text(json.dumps(dict(status='PASS',scope='source equivalence except schedule/checkpoint migration; original schedule hash; all schedule values; actual stop block; describe and launcher dry-run',experiments=report),indent=2))
print('PASS: prefix, schedule boundaries, exact update counts, unchanged dynamics, launcher')
