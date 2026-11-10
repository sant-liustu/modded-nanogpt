"""Figure 7 reproduction. Python >=3.10, PyTorch >=2.0, single GPU or torchrun DDP."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--target', type=Path, help='Reference run directory; sets WD=0')
    p.add_argument('--precision', choices=['float32', 'bfloat16'], default='bfloat16')
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--micro-batch', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--eval-examples', type=int, default=1000)
    p.add_argument('--save-interval', type=int, default=5000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--strict-fp32', action='store_true', help='Disable upstream TF32 defaults')
    p.add_argument('--device', default='cuda')
    p.add_argument('--tiny', action='store_true', help='Small synthetic-data smoke test only')
    p.add_argument('--gamma-cosine', action='store_true',
                   help='Gamma uses original schedule; ELR norm excludes gamma')
    a = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world > 1
    if distributed:
        if a.device.startswith('cuda'):
            torch.cuda.set_device(local_rank)
            a.device = f'cuda:{local_rank}'
        dist.init_process_group(backend='nccl' if a.device.startswith('cuda') else 'gloo')
    master = rank == 0
    if a.steps <= 400 and not a.tiny:
        p.error('Formal runs need more than 400 steps of warmup')
    if a.micro_batch <= 0 or a.batch_size % (a.micro_batch * world):
        p.error('global batch-size must be divisible by micro-batch * world size')
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / 'latest.pt'
    if (a.out / 'metrics.csv').exists() and not a.resume:
        p.error('Output exists; choose a new directory or --resume')
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = not a.strict_fp32
    torch.backends.cudnn.allow_tf32 = not a.strict_fp32
    cfg = dict(n_layer=12, n_head=12, n_embd=768, block_size=256,
               vocab_size=50304, bias=False, learnable_ln=True,
               weight_tying=True, dropout=0.0, init_scale=0.02)
    if a.tiny:
        cfg.update(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=128)
    model = GPT(GPTConfig(**cfg)).to(a.device)
    initial_hash = hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes()
                                         for x in model.parameters())).hexdigest()
    wd = 0.0 if a.target else a.weight_decay
    optimizer = model.configure_optimizers('adam', wd, 6e-4, (0.9, 0.95),
                                          a.device.split(':')[0],
                                          ['wte', 'wpe', 'mlp', 'attn', 'lm_head'], 0)
    gamma_names = [name for name, _ in model.named_parameters()
                   if name.endswith('.weight') and
                   ('.ln_1.' in name or '.ln_2.' in name or '.ln_f.' in name)]
    if a.gamma_cosine:
        gamma_ids = {id(param) for name, param in model.named_parameters() if name in gamma_names}
        assert gamma_ids == {id(param) for param in optimizer.param_groups[1]['params']}
        optimizer.param_groups[0]['role'] = 'matrix'
        optimizer.param_groups[1]['role'] = 'gamma'
    # Independent generators preserve training batches across eval and precision variants.
    train_rng = torch.Generator().manual_seed(a.seed + 1000)
    eval_rng = torch.Generator().manual_seed(a.seed + 2000)
    data = {s: np.memmap(a.data_dir / (s + '.bin'), dtype=np.uint16, mode='r')
            for s in ['train', 'val']}
    for s, d in data.items():
        if len(d) <= cfg['block_size']:
            raise ValueError(f'{s}.bin is too short')
    metadata = dict(config=cfg, seed=a.seed, steps=a.steps, batch_size=a.batch_size,
                    micro_batch=a.micro_batch, initial_hash=initial_hash,
                    data_sizes={s: len(d) for s, d in data.items()},
                    precision=a.precision, tf32=not a.strict_fp32,
                    weight_decay=wd, target=str(a.target), torch_version=torch.__version__,
                    norm='upstream model.weight_norm(), before optimizer step')
    if distributed:
        metadata.update(world_size=world, sampling='shared global indices partitioned by rank')
    if a.gamma_cosine:
        metadata.update(gamma_cosine=True, gamma_names=gamma_names,
                        norm='global L2 excluding LayerNorm gamma, before optimizer step')
    target = None
    if a.target:
        ref = json.loads((a.target / 'config.json').read_text())
        for key in ['config', 'seed', 'steps', 'batch_size', 'micro_batch', 'initial_hash', 'data_sizes']:
            if ref[key] != metadata[key]:
                raise ValueError(f'Reference mismatch: {key}')
        if ref['norm'] != metadata['norm'] or ref.get('gamma_cosine', False) != a.gamma_cosine:
            raise ValueError('Reference norm/gamma schedule mismatch')
        if ref.get('world_size', 1) != world:
            raise ValueError('Reference world size mismatch')
        target = np.load(a.target / 'elr.npy')
        if len(target) != a.steps or not np.isfinite(target).all():
            raise ValueError('Reference trajectory incomplete or nonfinite')
    start = 0
    trajectory = np.full(a.steps, np.nan)
    if a.resume:
        state = torch.load(ckpt, map_location=a.device, weights_only=False)
        if state['metadata'] != metadata:
            raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        train_rng.set_state(state['train_rng'].cpu())
        rng_state = state['rank_rng'][rank] if distributed else state
        torch.set_rng_state(rng_state['torch_rng'].cpu())
        if a.device.startswith('cuda'):
            if distributed:
                torch.cuda.set_rng_state(rng_state['cuda_rng'].cpu(), device=a.device)
            else:
                torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        trajectory = state['trajectory']
        start = state['step']
        # Drop rows newer than the last committed checkpoint before replay.
        for name in (['metrics.csv', 'eval.csv'] if master else []):
            path = a.out / name
            if path.exists():
                with path.open(newline='') as f:
                    reader = csv.DictReader(f)
                    fields = reader.fieldnames
                    rows = [r for r in reader if int(r['step']) < start]
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
    if master:
        (a.out / 'config.json').write_text(json.dumps(metadata, indent=2))
    forward = torch.compile(model) if a.compile else model
    if distributed:
        forward = DDP(forward, device_ids=[local_rank] if a.device.startswith('cuda') else None)
    if master:
        print(f'world_size={world} microbatch_per_rank={a.micro_batch} global_batch={a.batch_size} accumulation={a.batch_size // (a.micro_batch * world)}', flush=True)

    def batch(split, rng):
        d = data[split]
        count = a.micro_batch * world if split == 'train' and rng is train_rng else a.micro_batch
        ix = torch.randint(len(d) - cfg['block_size'], (count,), generator=rng)
        if count != a.micro_batch:
            ix = ix[rank * a.micro_batch:(rank + 1) * a.micro_batch]
        arr = np.stack([np.array(d[int(i):int(i)+cfg['block_size']+1], dtype=np.int64) for i in ix])
        return (torch.from_numpy(arr[:, :-1].copy()).to(a.device),
                torch.from_numpy(arr[:, 1:].copy()).to(a.device))

    def autocast():
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if a.precision == 'bfloat16' else contextlib.nullcontext())

    def append(name, row):
        if not master:
            return
        path = a.out / name
        new = not path.exists()
        with path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def save(step):
        rank_rng = None
        if distributed:
            local_rng = dict(torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state(a.device) if a.device.startswith('cuda') else [])
            rank_rng = [None] * world
            dist.all_gather_object(rank_rng, local_rng)
        if not master:
            return
        state = dict(step=step, metadata=metadata, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), trajectory=trajectory,
                     train_rng=train_rng.get_state(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [])
        if distributed:
            state['rank_rng'] = rank_rng
        torch.save(state, a.out / 'latest.tmp')
        os.replace(a.out / 'latest.tmp', ckpt)
        np.save(a.out / 'elr.npy', trajectory)

    for step in range(start, a.steps):
        if step % a.eval_interval == 0:
            model.eval()
            eval_rng.manual_seed(a.seed + 2000)
            row = dict(step=step)
            with torch.no_grad():
                for split in ['train', 'val']:
                    losses = []
                    for _ in range(math.ceil(a.eval_examples / a.micro_batch)):
                        x, y = batch(split, eval_rng)
                        with autocast():
                            _, loss = model(x, y)
                        losses.append(loss.item())
                    row[split + '_loss'] = sum(losses) / len(losses)
            append('eval.csv', row)
            model.train()
        if a.gamma_cosine:
            with torch.no_grad():
                norm = sum(param.square().sum() for name, param in model.named_parameters()
                           if name not in gamma_names).sqrt().item()
                gamma_norm = sum(param.square().sum() for name, param in model.named_parameters()
                                 if name in gamma_names).sqrt().item()
        else:
            norm = model.weight_norm().item()
        warmup = 4 if a.tiny else 400
        if step < warmup:
            base_lr = 6e-4 * step / warmup
        else:
            ratio = (step - warmup) / (a.steps - warmup)
            base_lr = 6e-5 + 0.5 * (1 + math.cos(math.pi * ratio)) * (6e-4 - 6e-5)
        lr = float(target[step]) * norm if target is not None else base_lr
        if not math.isfinite(lr) or not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError(f'Invalid control at step {step}')
        for group in optimizer.param_groups:
            group['lr'] = base_lr if a.gamma_cosine and group.get('role') == 'gamma' else lr
        trajectory[step] = lr / norm
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        accumulation = a.batch_size // (a.micro_batch * world)
        for _ in range(accumulation):
            x, y = batch('train', train_rng)
            with autocast():
                _, loss = forward(x, y)
                scaled = loss / accumulation
            loss_sum += loss.item() / accumulation
            scaled.backward()
        if distributed:
            averaged_loss = torch.tensor(loss_sum, device=a.device, dtype=torch.float64)
            dist.all_reduce(averaged_loss)
            loss_sum = averaged_loss.item() / world
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if not math.isfinite(loss_sum):
            raise FloatingPointError(f'Nonfinite loss at {step}')
        optimizer.step()
        metric = dict(step=step, loss=loss_sum, lr=lr, norm=norm,
                                  elr=trajectory[step], target_elr=float(target[step]) if target is not None else '',
                                  grad_norm=grad_norm.item())
        if a.gamma_cosine:
            metric.update(gamma_lr=base_lr, gamma_norm=gamma_norm)
        append('metrics.csv', metric)
        if master and step % 20 == 0:
            print(f'{step}/{a.steps} loss={loss_sum:.6f} lr={lr:.8g} norm={norm:.5f}', flush=True)
        if (step + 1) % a.save_interval == 0 or step + 1 == a.steps:
            save(step + 1)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
