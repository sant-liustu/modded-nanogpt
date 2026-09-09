"""Compare optimized LCA with its pre-optimization implementation.

Run from repository root: python experiments/eta_lambda_invariance/smoke_lca_cached_scalar_reduce.py
Uses CUDA for a small GPT check if available, plus two real CPU/Gloo DDP ranks.
The baseline is read from commit 13c20d2; no training entrypoint is imported.
"""
import ast
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / 'train_gpt2_mupp_w256_lr0p0036_wsd_wd0p033333_lrnormmatch_w768lr0p0036_wd0p1_lca.py'


class DistAudit:
    ReduceOp = dist.ReduceOp

    def __init__(self):
        self.sizes = []

    def all_reduce(self, tensor, op):
        self.sizes.append(tensor.numel())
        dist.all_reduce(tensor, op=op)


def definitions(source, device, distributed, audit):
    names = {'Rotary', 'apply_rotary_emb', 'RMSNorm', 'CausalSelfAttention',
             'MLP', 'Block', 'GPTConfig', 'GPT', 'LCADiagnostic'}
    namespace = dict(torch=torch, nn=nn, F=F, math=math, os=os, json=json,
                     dataclass=dataclass, contextlib=contextlib, device=device,
                     ctx=torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext(),
                     use_ddp=distributed, ddp_world_size=2 if distributed else 1,
                     dist=audit, B=2, T=8,
                     args=SimpleNamespace(simpson3_every=2, lca_fraction_eps=1e-12, lca_include_vectors=1))
    tree = ast.parse(source)
    exec(compile(ast.Module(body=[n for n in tree.body if getattr(n, 'name', None) in names],
                            type_ignores=[]), str(RUNNER), 'exec'), namespace)
    return namespace


def run_check(rank, distributed, directory, init_file=None):
    torch.set_num_threads(1)
    if distributed:
        dist.init_process_group('gloo', init_method=Path(init_file).as_uri(), rank=rank, world_size=2)
    try:
        current = RUNNER.read_text(encoding='utf-8')
        baseline = subprocess.check_output(
            ['git', 'show', '13c20d2:' + RUNNER.relative_to(ROOT).as_posix()], cwd=ROOT).decode('utf-8')
        device = 'cpu' if distributed or not torch.cuda.is_available() else 'cuda'
        audits = [DistAudit(), DistAudit()]
        ns = [definitions(s, device, distributed, a) for s, a in zip((baseline, current), audits)]
        torch.manual_seed(17)
        raw = ns[0]['GPT'](ns[0]['GPTConfig'](vocab_size=64, n_layer=2, n_head=2, n_embd=32)).to(device)
        copy = ns[1]['GPT'](ns[1]['GPTConfig'](vocab_size=64, n_layer=2, n_head=2, n_embd=32)).to(device)
        copy.load_state_dict(raw.state_dict())
        models = [torch.compile(m, backend='eager') for m in (raw, copy)]
        if distributed:
            models = [DDP(m) for m in models]
        # Different probes per rank, two microbatches: averaging must be correct.
        generator = torch.Generator(device=device).manual_seed(71 + rank)
        probes = [(torch.randint(0, 64, (2, 8), generator=generator, device=device),
                   torch.randint(0, 64, (2, 8), generator=generator, device=device)) for _ in range(2)]
        diagnostics = []
        counts = [0, 0]
        for i in range(2):
            logdir = Path(directory) / f'{distributed}_{rank}_{i}'
            logdir.mkdir()
            d = ns[i]['LCADiagnostic'](models[i], (raw, copy)[i], probes, str(logdir), True)
            evaluate = d._evaluate
            def counted(*args, _i=i, _evaluate=evaluate, **kwargs):
                counts[_i] += 1
                return _evaluate(*args, **kwargs)
            d._evaluate = counted
            diagnostics.append(d)
        opt = torch.optim.AdamW(raw.parameters(), lr=0.0001, weight_decay=0)
        target_rms = {n: p.detach().square().mean().sqrt() for n, p in raw.named_parameters() if p.ndim == 2}
        for step in range(1, 7):
            # Identical training data on both ranks; diagnostic probes differ.
            x = torch.arange(16, device=device).reshape(2, 8)
            with ns[0]['ctx']:
                _, loss = models[0](x, (x + 1) % 64, return_logits=False)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for n, p in raw.named_parameters():
                    if p.ndim == 2:
                        p.mul_(target_rms[n] / p.square().mean().sqrt())
            copy.load_state_dict(raw.state_dict())
            endpoint = {n: p.detach().clone() for n, p in raw.named_parameters()}
            for d, m in zip(diagnostics, (raw, copy)):
                d.maybe_run(step)
                assert all(torch.equal(p, endpoint[n]) and p.grad is None for n, p in m.named_parameters())
                assert d.model.training
        records = [[json.loads(line) for line in Path(d.history_path).read_text().splitlines()] for d in diagnostics]
        assert len(records[0]) == len(records[1])
        max_difference = 0.0
        for a, b in zip(*records):
            assert (a['step_end'], a['parameter_name']) == (b['step_end'], b['parameter_name'])
            keys = ['attribution', 'probe_loss_start', 'probe_loss_end', 'sum_attribution', 'residual']
            if a['record_type'] == 'global':
                keys += ['cumulative_exact_loss_change', 'cumulative_sum_attribution', 'cumulative_residual']
            for key in keys:
                difference = abs(a[key] - b[key])
                max_difference = max(max_difference, difference)
                assert math.isclose(a[key], b[key], rel_tol=2e-4, abs_tol=2e-6), (key, a[key], b[key])
        assert counts == [9, 7], counts
        nparams = len(list(raw.parameters()))
        if distributed:
            assert audits[1].sizes == [nparams + 2] * 3, audits[1].sizes
            assert sum(audits[0].sizes) > sum(audits[1].sizes)
        # A failed interval must restore parameters and keep the previous cache.
        d = diagnostics[1]
        cached = d.endpoint_cache
        evaluate = d._evaluate
        def fail(*a, **k):
            # Fail after mutating live parameters, not just before evaluation.
            with torch.no_grad():
                next(copy.parameters()).add_(1.0)
            raise RuntimeError('injected failure')
        d._evaluate = fail
        try:
            d.run('simpson3', 8)
            raise AssertionError('failure was not raised')
        except RuntimeError as error:
            assert str(error) == 'injected failure'
        finally:
            d._evaluate = evaluate
        assert d.endpoint_cache is cached and d.snapshot_steps['simpson3'] == 6
        assert all(torch.equal(p, endpoint[n]) and p.grad is None for n, p in copy.named_parameters())
        result = dict(rank=rank, distributed=distributed, device=device,
                      evaluations=counts, max_absolute_difference=max_difference,
                      baseline_reduce_elements=sum(audits[0].sizes),
                      optimized_reduce_elements=sum(audits[1].sizes),
                      optimized_reduce_calls=len(audits[1].sizes), result='PASS')
        Path(directory, f'result_{distributed}_{rank}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        run_check(0, False, directory)
        mp.spawn(run_check, args=(True, directory, str(Path(directory) / 'gloo_init')), nprocs=2, join=True)
