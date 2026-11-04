from pathlib import Path
import ast,json,hashlib,math,subprocess,sys,tempfile
from types import SimpleNamespace
root=Path(__file__).resolve().parent
source=root/'train_intrinsic_pulse.py'
t=ast.parse(source.read_text(encoding='utf-8'))
provenance=json.loads((root/'provenance.json').read_text())
allowed={'shared_elr','capture_training_state','restore_training_state'}
actual={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in allowed}
assert actual==provenance['inherited_definition_sha256']
for p in root.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
ns={'args':SimpleNamespace(warmup_iters=924,peak_elr=.06)}
exec(compile(ast.Module(body=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='shared_elr')],type_ignores=[]),'<schedule>','exec'),ns)
setup=t.body[next(i for i,n in enumerate(t.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='cli' for v in n.targets))+1:]
# The real setup assignments are executed, with an explicit original checkpoint identity.
assignments=[]
for n in setup:
    if isinstance(n,ast.If):break
    assignments.append(n)
loop=next(n for n in t.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
save=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='save_checkpoint')
report=[]
for start in (5000,10000,15000,16000):
    for arm,width,eta in [('baseline',0,.06),('elr005',24,.05),('elr004',30,.04),('elr003',40,.03),('elr002',60,.02),('elr001',120,.01)]:
        cli=SimpleNamespace(fork_step=start,experiment=arm)
        ns['cli']=cli
        exec(compile(ast.Module(body=assignments,type_ignores=[]),'<actual-cli-setup>','exec'),ns)
        updates=520 if not width else width+500
        assert cli.stage_start==start and cli.pulse_updates==width and cli.stop_step==start+updates
        values=[ns['shared_elr'](u) for u in range(start+1,cli.stop_step+1)]
        assert values==[eta]*width+[.06]*(updates-width)
        assert math.isclose(sum(values),31.2,rel_tol=1e-13)
        if width:assert math.isclose(width*eta,1.2,rel_tol=1e-14)
        cli.stop_after_save=False;cli.save_at=None
        saved=[];actual=[]
        control={'cli':cli,'x':None,'y':None,'save_fork_checkpoint':lambda step,x,y:saved.append(step),'saved':saved,'actual':actual,'start':start,'end':cli.stop_step,'checkpoints':{start+width,cli.stop_step} if width else {cli.stop_step}}
        body=ast.parse('save_checkpoint = step in checkpoints').body+[save]+ast.parse('actual.append(step+1)').body
        test=ast.For(target=ast.Name(id='step',ctx=ast.Store()),iter=ast.parse('range(start,end+2)',mode='eval').body,body=body,orelse=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[test],type_ignores=[])),'<actual-stop-block>','exec'),control)
        assert actual==list(range(start+1,cli.stop_step+1))
        assert saved==sorted(control['checkpoints'])
        report.append(dict(fork_step=start,arm=arm,pulse_updates=width,recovery_updates=(500 if width else 0),updates=updates,intrinsic_time=sum(values),stop=cli.stop_step))
assert any(isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is True for n in loop.body)
assert 'val_loader.reset()' in source.read_text()
with tempfile.TemporaryDirectory(dir=root) as tmp:
    checkpoint_paths=[]
    for step in (5000,10000,15000,16000):
        p=Path(tmp)/str(step);p.mkdir();checkpoint_paths.append(str(p))
        files=['rank00000.pt','rank00001.pt']
        (p/'complete.json').write_text(json.dumps(dict(version=1,step=step,world_size=2,files=files)))
        for f in files:(p/f).touch()
    cmd=[sys.executable,str(root/'run_probes.py'),'--checkpoints',*checkpoint_paths,'--input-bin','/data/train*.bin','--input-val-bin','/data/val*.bin','--output',str(Path(tmp)/'out'),'--dry-run']
    jobs=json.loads(subprocess.check_output(cmd,text=True));assert len(jobs)==24
    for step in (5000,10000,15000,16000):
        arms=[j for j in jobs if j['fork_step']==step];assert len(arms)==6
        assert len({j['input_checkpoint'] for j in arms})==1
    assert subprocess.run(cmd+['--probe-val-tokens','1'],capture_output=True).returncode!=0
budget=json.loads(subprocess.check_output([sys.executable,str(source),'--fork-step','5000','--experiment','elr001','--describe-only'],text=True))
assert budget['continuation_updates']==620 and budget['validation_every']==1 and budget['validation_tokens']==1048576
(root/'verification.json').write_text(json.dumps(dict(status='PASS',scope='unchanged model/optimizer/projection/loader functions; actual CLI setup and stop block all24 schedules; same-input launcher dependencies; initial/every-step fixed-prefix validation; describe budget',experiments=report),indent=2))
print('PASS:24 schedules/stop boundaries, source equivalence, launcher and validation')
