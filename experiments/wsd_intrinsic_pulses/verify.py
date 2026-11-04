from pathlib import Path
import argparse,ast,json,hashlib,math,subprocess,sys,tempfile
from types import SimpleNamespace
option_parser=argparse.ArgumentParser(description='Verify source, all24 schedules and launcher; optional dependency-free static mode')
option_parser.add_argument('--static-only',action='store_true',help='run all stdlib checks without importing the torch-based trainer')
verify_options=option_parser.parse_args()
root=Path(__file__).resolve().parent
source=root/'train_intrinsic_pulse.py'
t=ast.parse(source.read_text(encoding='utf-8'))
provenance=json.loads((root/'provenance.json').read_text())
allowed={'shared_elr','capture_training_state','restore_training_state'}
def canonical(node):
    if isinstance(node, ast.AST):
        fields=[]
        for key,value in ast.iter_fields(node):
            # Added in3.12; absent in3.10. Only empty generic parameter lists
            # are equivalent to absence. Nonempty lists remain in the hash.
            if key=='type_params' and value==[]:
                continue
            fields.append((key,canonical(value)))
        return (type(node).__name__,tuple(sorted(fields)))
    if isinstance(node,list):
        return tuple(canonical(value) for value in node)
    return node

def definition_hash(node):
    # Includes decorators, defaults, annotations, nesting, f-string expressions
    # and docstrings; excludes source positions/comments and empty type_params.
    return hashlib.sha256(repr(canonical(node)).encode()).hexdigest()

text=source.read_text(encoding='utf-8')
actual={n.name:definition_hash(n) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in allowed}
expected=provenance['inherited_normalized_ast_sha256']
mismatches=sorted(k for k in actual.keys() | expected.keys() if actual.get(k)!=expected.get(k))
assert not mismatches, 'Inherited source differs from reference: '+', '.join(mismatches)
# feature_version alone does not emulate an older AST schema: strip the actual
# added field too, then compare every inherited definition (including f-strings).
legacy=ast.parse(text,feature_version=(3,10))
for node in ast.walk(legacy):
    if hasattr(node,'type_params'):
        assert node.type_params==[]
        delattr(node,'type_params')
        node._fields=tuple(k for k in node._fields if k!='type_params')
assert actual=={n.name:definition_hash(n) for n in legacy.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in allowed}
base="@torch.no_grad()\ndef f(x=1):\n    return f'value={x!r:>4}'\n"
a=ast.parse(base).body[0]
for changed in (base.replace('no_grad','enable_grad'),base.replace('x=1','x=2'),base.replace('value=','other='),base.replace('x!r','(x+1)!r'),base.replace('>4','>5')):
    assert definition_hash(a)!=definition_hash(ast.parse(changed).body[0])
assert definition_hash(a)==definition_hash(ast.parse(base.replace('return', '# note\n    return')).body[0])
# A real logic/nesting change must still be rejected.
assert definition_hash(ast.parse('def f(x):\n    if x:\n        return 1\n    return 2\n').body[0])!=definition_hash(ast.parse('def f(x):\n    if x:\n        return 1\n        return 2\n').body[0])
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
if not verify_options.static_only:
    budget=json.loads(subprocess.check_output([sys.executable,str(source),'--fork-step','5000','--experiment','elr001','--describe-only'],text=True))
    assert budget['continuation_updates']==620 and budget['validation_every']==1 and budget['validation_tokens']==1048576
(root/'verification.json').write_text(json.dumps(dict(status='PASS',python=sys.version,static_only=verify_options.static_only,describe_checked=not verify_options.static_only,scope='unchanged model/optimizer/projection/loader functions; actual CLI setup and stop block all24 schedules; same-input launcher dependencies; initial/every-step fixed-prefix validation; describe budget',experiments=report),indent=2))
print('PASS:24 schedules/stop boundaries, source equivalence, launcher and validation')
