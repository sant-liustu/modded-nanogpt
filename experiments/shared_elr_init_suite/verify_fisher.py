"""Real FineWeb paired CUDA runs: measurement must not alter training state."""
from pathlib import Path
import json, os, subprocess, sys, time
import numpy as np
import torch

root = Path(__file__).parent
runner = os.environ.get('FISHER_TEST_RUNNER', 'train_muon.py')
lr_mode = os.environ.get('FISHER_TEST_LR_MODE', 'rms_elr')
training_arm = os.environ.get('FISHER_TEST_ARM', 'fanin_fixed')
if lr_mode == 'absolute' and runner != 'train.py':
    raise ValueError('absolute LR is supported by train.py only')
if runner not in ('train.py', 'train_muon.py'):
    raise ValueError('FISHER_TEST_RUNNER must be train.py or train_muon.py')
out = root/'local_checks'/('fisher_'+Path(runner).stem+'_'+time.strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True)
width = int(os.environ.get('FISHER_TEST_WIDTH', '128'))
depth = int(os.environ.get('FISHER_TEST_DEPTH', '1'))
steps = int(os.environ.get('FISHER_TEST_STEPS', '4'))
length = int(os.environ.get('FISHER_TEST_LENGTH', '32'))
micro = int(os.environ.get('FISHER_TEST_MICRO_BATCH', '2'))
accumulation = int(os.environ.get('FISHER_TEST_ACCUMULATION', '2'))
interval = max(1, steps//2) if steps <= 4 else max(1, steps//5)
states, timings, records = {}, {}, []
for arm, every in [('off', 0), ('on', interval)]:
    cmd = [sys.executable, str(root/runner), '--arm',training_arm,
           '--model-width',str(width),'--model-depth',str(depth),'--expected-world-size','1',
           '--batch-size','1','--device-batch-size','1','--sequence-length',str(length),
           '--vocab-size','50304','--num-iterations',str(steps),'--warmup-iters',str(max(1,steps//10)),'--warmdown-iters','1',
           '--compile-model','0','--val-tokens',str(length),'--schedule','cosine',
           '--input-bin','F:/datasets/fineweb10B-gpt2/raw/fineweb_train_000001.bin',
           '--input-val-bin','F:/datasets/fineweb10B-gpt2/raw/fineweb_val_000000.bin',
           '--log-root',str(out/arm/'logs'),'--run-name',arm,
           '--checkpoint-dir',str(out/arm/'checkpoints'),'--peak-elr','0.1',
           '--save-at',str(max(1,steps//2)),
           '--fisher-every',str(every),'--fisher-draws','2','--fisher-sequence-length',str(length),
           '--fisher-batch-size',str(micro),'--fisher-accumulation-steps',str(accumulation),
           '--fisher-token-chunk-size','4']
    if lr_mode == 'absolute':
        cmd += ['--lr-mode','absolute','--learning-rate','0.0009','--embedding-lr-multiplier','2','--min-lr-ratio','0.1',
                '--weight-decay','0.1','--gamma-weight-decay',str(0.1 if training_arm=='fanin_wd' else 0.0),'--grad-clip','0']
    if arm == 'off':
        # Exercise automatic final full-checkpoint output without explicit CLI path.
        for option in ('--checkpoint-dir','--save-at'):
            index=cmd.index(option)
            del cmd[index:index+2]
    (out/f'{arm}_command.json').write_text(json.dumps(cmd,indent=2))
    start=time.perf_counter()
    with (out/f'{arm}.log').open('w') as f:
        proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    timings[arm]=time.perf_counter()-start
    if proc.returncode:
        print((out/f'{arm}.log').read_text()[-8000:]);raise SystemExit(proc.returncode)
    checkpoint=(out/arm/f'checkpoints/step{steps:06d}/rank00000.pt' if arm=='on'
                else next((out/arm/'logs').glob(f'*/checkpoints/step{steps:06d}/rank00000.pt')))
    states[arm]=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if every:
        path=next((out/arm/'logs').glob('*/fisher_history.jsonl'))
        records=[json.loads(l) for l in path.read_text().splitlines()]

def equal(a,b,path=''):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,path

for key in ('model','optimizers','schedulers','rng','train_loader','val_loader','next_batch','targets'):
    equal(states['off'][key],states['on'][key],key)
assert [r['step'] for r in records]==sorted(set(range(0,steps+1,interval))|{steps})
assert len({r['input_sha256'] for r in records})==1
for r in records:
    snapshot_path=path.parent/r['model_checkpoint']
    snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=False)
    assert snapshot['format']=='fisher_model_only_v1' and snapshot['step']==r['step']
    assert snapshot['input_sha256']==r['input_sha256']
    assert snapshot['fisher_initial_rms']==states['on']['fisher_initial_rms']
    assert not any(k in snapshot for k in ('optimizers','schedulers','rng','train_loader'))
    assert not snapshot_path.with_suffix('.tmp').exists()
    if r['step']==steps: equal(snapshot['model'],states['on']['model'],'snapshot/model')
    assert r['vocabulary_head']=='chunked_hidden_chain_rule' and r['token_chunk_size']==4
    assert r['precision']=='bf16_autocast_fp32_accumulation_fp64_square_sum'
    assert set(r['groups'])=={'Emb','QK','VO','FFN','Norm'}
    assert r['valid_tokens']==micro*accumulation*length
    assert r['sequences']==micro*accumulation and r['accumulation_steps']==accumulation and r['micro_batch_size']==micro
    assert r['draws']==2 and r['draw_seeds']==[1729,1730]
    for v in list(r['groups'].values())+list(r['parameters'].values()):
        a,b=v['mean_fisher_draws']
        assert abs(v['mean_fisher']-(a+b)/2)<1e-15
        assert abs(v['mc_standard_error']-abs(a-b)/2)<1e-15
    for group,v in r['groups'].items():
        ps=[p for p in r['parameters'].values() if p['group']==group]
        assert sum(p['numel'] for p in ps)==v['numel']
        assert abs(sum(p['trace'] for p in ps)-v['trace'])<1e-8*max(1,v['trace'])
    assert all(p['mean_fisher']>=0 for p in r['parameters'].values())
    for name, p in r['parameters'].items():
        initial = records[0]['parameters'][name]
        assert p['initial_rms'] == initial['rms'] == initial['initial_rms']
        assert p['initial_rms'] == states['on']['fisher_initial_rms'][name]
        assert p['shape'] == list(states['on']['model'][name].shape)
        tensor=snapshot['model'][name].double()
        assert abs(tensor.square().mean().sqrt().item()-p['rms'])<1e-12
        if r['step'] == steps:
            tensor = states['on']['model'][name].double()
            expected = tensor.square().mean().sqrt().item()
            assert abs(p['rms']-expected) <= 1e-12*max(1, expected)

# Load into a fresh model without executing the trainer's top-level loop.
import ast, math
from dataclasses import dataclass
from types import SimpleNamespace
source=ast.parse(snapshot['source_code'])
model_names={'Rotary','apply_rotary_emb','RMSNorm','CausalSelfAttention','MLP','Block','GPTConfig','GPT'}
model_ast=ast.Module(body=[n for n in source.body
                          if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in model_names], type_ignores=[])
namespace=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math,dataclass=dataclass,
               args=SimpleNamespace(**snapshot['config']),__name__='__main__')
exec(compile(model_ast,runner,'exec'),namespace)
restored_model=namespace['GPT'](namespace['GPTConfig'](**snapshot['model_config']))
restored_model.load_state_dict(snapshot['model'],strict=True)
equal(restored_model.state_dict(),snapshot['model'],'fresh_model_load')
assert restored_model.transformer.wte.weight is restored_model.lm_head.weight
with torch.no_grad():
    logits,_=restored_model(torch.zeros((1,2),dtype=torch.long),torch.zeros((1,2),dtype=torch.long))
    assert torch.isfinite(logits).all()
del restored_model

# Resume the same on-run from its intermediate checkpoint, preserving step-zero RMS.
resume_cmd = cmd.copy()
resume_cmd[resume_cmd.index('--log-root')+1] = str(out/'resumed'/'logs')
resume_cmd[resume_cmd.index('--checkpoint-dir')+1] = str(out/'resumed'/'checkpoints')
resume_cmd[resume_cmd.index('--save-at')+1] = str(steps)
resume_cmd += ['--resume', str(out/'on'/f'checkpoints/step{max(1,steps//2):06d}')]
with (out/'resumed.log').open('w') as f:
    proc = subprocess.run(resume_cmd, stdout=f, stderr=subprocess.STDOUT,
                          env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
if proc.returncode:
    print((out/'resumed.log').read_text()[-8000:]); raise SystemExit(proc.returncode)
resumed = torch.load(out/'resumed'/f'checkpoints/step{steps:06d}/rank00000.pt',
                     map_location='cpu', weights_only=False)
for key in ('model','optimizers','schedulers','rng','train_loader','val_loader','next_batch','targets','fisher_initial_rms'):
    equal(states['on'][key], resumed[key], 'resumed/'+key)
resume_records = [json.loads(line) for line in next((out/'resumed'/'logs').glob('*/fisher_history.jsonl')).read_text().splitlines()]
for record in resume_records:
    for name, p in record['parameters'].items():
        assert p['initial_rms'] == states['on']['fisher_initial_rms'][name]
if lr_mode == 'absolute':
    history=[json.loads(line) for line in (path.parent/'tensor_rms_elr_history.jsonl').read_text().splitlines()]
    schedule=json.loads((path.parent/'schedule.json').read_text())
    assert schedule['kind']=='absolute_lr'
    rates={row['update_step']:row['learning_rate'] for row in schedule['steps']}
    for row in history:
        assert row['lr_policy']=='baseline_absolute_lr'
        assert row['lr']==rates[row['step']]*(2 if row['name']=='transformer.wte.weight' else 1)
        assert row['target_rms_elr'] is None
    warm=max(1,steps//10)
    expected_final=0.0009*(0.1+0.9*0.5*(1+math.cos(math.pi*(steps-1-warm)/(steps-warm))))
    assert abs(rates[steps]-expected_final)<1e-15
    assert len({row['pre_update_rms'] for row in history})>1
    for opt in states['on']['optimizers']:
        for group in opt['param_groups']:
            assert group['weight_decay'] in (0.0,0.1) and group['betas']==(0.9,0.95) and group['eps']==1e-8
    metadata=json.loads((path.parent/'schedule_metadata.json').read_text())
    assert metadata['fixed_norm']==(training_arm=='fanin_fixed')
    if training_arm=='fanin_fixed':
        norms=[json.loads(line) for line in (path.parent/'norm_control_history.jsonl').read_text().splitlines()]
        assert norms and max(row['relative_error'] for row in norms)<1e-6
    else:
        assert metadata['gamma_wd']==0.1
        assert all(g['weight_decay']==0.1 for opt in states['on']['optimizers'] for g in opt['param_groups'])
        assert not (path.parent/'norm_control_history.jsonl').exists()
        assert any(p['rms']!=p['initial_rms'] for p in records[-1]['parameters'].values() if len(p['shape'])==2)

report=dict(status='PASS',runner=runner,lr_mode=lr_mode,training_arm=training_arm,scope=f'single CUDA eager real FineWeb w{width} d{depth} vocab50304 B1 T{length}, {steps} updates',
            fisher_sequences=micro*accumulation, fisher_micro_batch_size=micro, fisher_accumulation_steps=accumulation, fisher_draws=2,
            model_snapshots_verified=True, default_final_full_checkpoint_verified=True,
            exact_training_state_equal=True, initial_and_current_rms_verified=True,
            exact_resume_state_equal=True, wall_seconds=timings,
            probe_seconds=[r['seconds'] for r in records],
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
            measurements=[dict(step=r['step'],groups=r['groups']) for r in records],output=str(out))
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='measurements'},indent=2))
