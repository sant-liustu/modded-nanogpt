"""Verify baseline equivalence, controlled differences, schedules and CPU updates."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'provenance.json').read_text())
source = Path(manifest[0]['source'])

def tree(path):
    return ast.parse(path.read_text(encoding='utf-8'))

def definitions(t):
    return {n.name: ast.dump(n) for n in t.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name != 'Hyperparameters'}

def config(t):
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == 'Hyperparameters')
    return {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)}

def load(t):
    names = {'Rotary', 'apply_rotary_emb', 'RMSNorm', 'CausalSelfAttention', 'MLP',
             'Block', 'GPTConfig', 'GPT', 'shared_elr', 'get_lr', 'build_fixed_norm_state',
             'apply_fixed_norm_control', 'project_fixed_rms_eager'}
    nodes = [n for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    ns = dict(torch=torch, nn=nn, F=F, math=math, dataclass=dataclass,
              args=SimpleNamespace(arm='fanin_fixed'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<extracted>', 'exec'), ns)
    ns['project_fixed_rms'] = ns['project_fixed_rms_eager']
    return ns

original = tree(source)
original_config = config(original)
reference_budget = json.loads(subprocess.check_output([sys.executable, str(source), '--describe-only'], text=True))
reference_ns = load(original)
report = []
for entry in manifest:
    path = ROOT / entry['script']
    t = tree(path)
    compile(t, str(path), 'exec')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    # Every function, model class, optimizer helper and projection is identical.
    assert {k:v for k,v in definitions(t).items() if k != 'shared_elr'} == {k:v for k,v in definitions(original).items() if k != 'shared_elr'}
    cfg = config(t)
    allowed_changes = {'model_width', 'model_depth', 'log_root', 'num_iterations',
                       'warmup_iters', 'warmdown_iters', 'target_tpp', 'reference_tpp',
                       'expected_world_size', 'device_batch_size', 'peak_elr', 'schedule'}
    assert {k:v for k,v in cfg.items() if k not in allowed_changes} == {
        k:v for k,v in original_config.items() if k not in allowed_changes}
    assert cfg['schedule'] == 'wsd'
    budget = json.loads(subprocess.check_output([sys.executable, str(path), '--describe-only'], text=True))
    assert budget['total_updates'] == 18858
    assert budget['stop_step'] == budget['stable_end'] == 15000
    assert budget['executed_tokens'] == 1966080000
    assert budget['warmup_updates'] == 924 and budget['warmdown_updates'] == 3858
    assert budget['expected_world_size'] * budget['device_batch_size'] == 128
    ns = load(t)
    with torch.device('meta'):
        model = ns['GPT'](ns['GPTConfig'](vocab_size=50304, n_embd=budget['width'],
                         n_head=budget['width']//128, n_layer=budget['depth']))
    assert sum(p.numel() for p in model.parameters()) == budget['parameter_count']
    assert model.transformer.wte.weight is model.lm_head.weight
    a = SimpleNamespace(**cfg)
    ns['args'] = a
    reference_ns['args'] = a
    for step in range(1, 18859):
        expected = .06*step/924 if step<=924 else (.06 if step<=15000 else .06*(18858-step)/3858)
        assert math.isclose(ns['shared_elr'](step), expected, rel_tol=1e-14, abs_tol=1e-16)
        assert ns['get_lr'](step-1) == reference_ns['get_lr'](step-1)
    assert ns['shared_elr'](924) == ns['shared_elr'](15000) == .06 and ns['shared_elr'](18858) == 0
    assert budget['gamma_warmup_updates'] == 924
    # Both trainers' tiny-model forward/backward, AdamW and fixed-RMS projection
    # must agree exactly under the same seed and three nontrivial updates.
    tiny = dict(vocab_size=128, n_embd=128, n_head=1, n_layer=1)
    pair = []
    for namespace in (reference_ns, ns):
        torch.manual_seed(17)
        m = namespace['GPT'](namespace['GPTConfig'](**tiny))
        fixed = namespace['build_fixed_norm_state'](m)
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0)
        pair.append((namespace, m, fixed, opt))
    torch.manual_seed(23)
    x = torch.randint(0, 128, (2, 8))
    for step in range(1, 4):
        losses = []
        for namespace, m, fixed, opt in pair:
            opt.zero_grad(set_to_none=True)
            _, loss = m(x, x)
            assert torch.isfinite(loss)
            losses.append(loss.detach())
            loss.backward()
            opt.step()
            namespace['apply_fixed_norm_control'](fixed, step)
        assert torch.equal(*losses)
        assert all(torch.equal(v, pair[1][1].state_dict()[k]) for k,v in pair[0][1].state_dict().items())
    for option in (['--tpp', '20'], ['--peak-elr', '0.1'], ['--batch-size', '64']):
        result = subprocess.run([sys.executable, str(path), '--describe-only', *option], capture_output=True)
        assert result.returncode != 0, option
    report.append(dict(script=entry['script'], budget=budget, non_schedule_definitions_equal=True,
                       all_step_wsd_schedule_checked=True, cpu_three_update_regression='bitwise PASS'))
jobs = json.loads(subprocess.check_output([sys.executable, str(ROOT/'launch.py'),
    '--train-glob', '/data/train_*.bin', '--val-glob', '/data/val_*.bin',
    '--output', str(ROOT/'outputs'/'dry_run'), '--dry-run'], text=True))
flat = [job for stages in jobs.values() for stage in stages for job in stage]
assert len(flat) == 5
for job in flat:
    command = job['command']
    ranks = int(command[command.index('--expected-world-size')+1])
    microbatch = int(command[command.index('--device-batch-size')+1])
    assert len(job['gpu_ids']) == ranks and ranks * microbatch == 128
(ROOT/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('PASS: five budgets, actual parameter counts, unchanged non-schedule definitions, all-step WSD schedules, CPU three-update regression, guards and launcher dry-run')

