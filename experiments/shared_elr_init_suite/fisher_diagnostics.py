"""Paper 2502.19002 Eq.3-4 / Sophia-G GNB diagnostics for this Muon runner."""
import math
import time
import torch
import torch.nn.functional as F


def block_type(name):
    if name in ('transformer.wte.weight', 'lm_head.weight'):
        return 'Emb'
    if 'norm' in name:
        return 'Norm'
    if '.attn.c_q.' in name or '.attn.c_k.' in name:
        return 'QK'
    if '.attn.c_v.' in name or '.attn.c_proj.' in name:
        return 'VO'
    if '.mlp.' in name:
        return 'FFN'
    raise ValueError(f'Unclassified Fisher parameter: {name}')


@torch.no_grad()
def parameter_rms(model):
    """RMS (not Frobenius norm), counted once for tied parameters."""
    return {n: torch.linalg.vector_norm(p.detach(), dtype=torch.float64).item()
            / math.sqrt(p.numel()) for n, p in model.named_parameters() if p.requires_grad}


def should_measure(step, start_step, final_step, every):
    return every > 0 and (step in (0, 100, start_step, final_step) or step % every == 0)


def probe(model, x, y, draws, seed, initial_rms=None, micro_batch_size=64):
    """Each independent draw accumulates token-mean gradients, THEN squares.

    Inputs may be CPU resident. y only selects valid positions and full logits.
    Private gradient buffers never write training parameter .grad.
    """
    if draws < 1 or micro_batch_size < 1:
        raise ValueError('Expected positive draws and micro-batch size')
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if initial_rms is not None and set(initial_rms) != {n for n, _ in params}:
        raise ValueError('Initial RMS parameter names do not match model')
    if any(p.dtype != torch.float32 for _, p in params):
        raise ValueError('Fisher diagnostic expects FP32 master parameters')
    mapping = {n: block_type(n) for n, _ in params}
    aliases = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(n)
    modes = [(m, m.training) for m in model.modules()]
    device = params[0][1].device
    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    before_bytes = torch.cuda.memory_allocated(device) if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    current_rms = parameter_rms(model)
    N = int((y != -1).sum().item())
    if not N:
        raise ValueError('No valid Fisher tokens')
    rows = []
    draw_seeds = [seed + draw for draw in range(draws)]
    try:
        model.eval()
        with torch.random.fork_rng(devices=[device.index] if cuda else []), torch.enable_grad(), torch.autocast(device.type, enabled=False):
            for draw_seed in draw_seeds:
                accum = [torch.zeros_like(p) for _, p in params]
                generator = torch.Generator(device=device).manual_seed(draw_seed)
                for offset in range(0, len(x), micro_batch_size):
                    xb = x[offset:offset+micro_batch_size].to(device)
                    yb = y[offset:offset+micro_batch_size].to(device)
                    if not (yb != -1).any():
                        continue
                    logits, real_loss = model(xb, yb, return_logits=True)
                    del real_loss
                    z = logits[yb != -1]
                    probs = z.detach().softmax(-1)
                    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
                    # Sum / global valid-token count handles unequal masks exactly.
                    loss = F.cross_entropy(z, sampled, reduction='sum') / N
                    grads = torch.autograd.grad(loss, [p for _, p in params])
                    for total, grad in zip(accum, grads):
                        total.add_(grad.detach())
                    del grads, loss, logits, z, probs, sampled, xb, yb
                traces = torch.stack([N*g.double().square().sum() for g in accum]).cpu().tolist()
                if not all(math.isfinite(t) for t in traces):
                    raise FloatingPointError('Nonfinite Fisher estimate')
                rows.append(dict(zip(mapping, traces)))
                del accum
    finally:
        for module, mode in modes:
            module.training = mode
    if cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    counts = {n: p.numel() for n, p in params}
    def stats(names):
        n = sum(counts[k] for k in names)
        values = [sum(row[k] for k in names)/n for row in rows]
        mean = sum(values)/draws
        se = (math.sqrt(sum((v-mean)**2 for v in values)/(draws-1)/draws)
              if draws > 1 else None)
        return dict(numel=n, mean_fisher=mean, trace=mean*n,
                    mc_standard_error=se, mean_fisher_draws=values)
    peak = torch.cuda.max_memory_allocated(device) if cuda else 0
    return dict(estimator='N_times_accumulated_pseudolabel_mean_gradient_squared',
                valid_tokens=N, sequences=len(x), sequence_length=x.shape[1],
                micro_batch_size=micro_batch_size,
                accumulation_steps=math.ceil(len(x)/micro_batch_size),
                draws=draws, seed=seed, draw_seeds=draw_seeds, seconds=elapsed, precision='fp32_forward_accumulation_fp64_square_sum',
                peak_allocated_bytes=peak, extra_peak_allocated_bytes=max(0,peak-before_bytes),
                uncertainty='pseudo-label Monte Carlo only, conditional on fixed inputs; two-draw SE is noisy' if draws > 1 else 'single estimate; Monte Carlo standard error unavailable',
                embedding_policy='tied embedding/head counted once as Emb',
                groups={g:stats([n for n in mapping if mapping[n]==g]) for g in sorted(set(mapping.values()))},
                norm_definition='sqrt(sum(parameter**2)/numel); no normalization applied to Fisher',
                parameters={n:dict(stats([n]), group=mapping[n], aliases=aliases[id(p)],
                                   shape=list(p.shape), rms=current_rms[n],
                                   initial_rms=None if initial_rms is None else initial_rms[n]) for n,p in params})
