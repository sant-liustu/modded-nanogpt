"""Fast runtime check for the per-tensor standard-AdamW implementation pattern."""

import math

import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    controlled = [
        torch.nn.Parameter(torch.randn(16 + (i % 4) * 8, 12 + (i % 3) * 4, device=device))
        for i in range(73)
    ]
    controlled_names = [f'w{i}' for i in range(72)] + ['transformer.wte.weight']
    gamma = torch.nn.Parameter(torch.ones(32, device=device))
    optimizer = torch.optim.AdamW(
        [
            dict(params=[parameter], lr=1.0, weight_decay=0.1, single_elr_name=name)
            for name, parameter in zip(controlled_names, controlled)
        ],
        betas=(0.9, 0.95),
        fused=device.type == 'cuda',
    )
    gamma_optimizer = torch.optim.AdamW(
        [dict(params=[gamma], weight_decay=0.0)],
        lr=0.0018,
        betas=(0.9, 0.95),
        fused=device.type == 'cuda',
    )

    target_elr = 0.017
    embedding_elr_multiplier = 2.0
    fro_norms = torch._foreach_norm(controlled, 2)
    rms_values = torch.stack([
        fro_norm / math.sqrt(parameter.numel())
        for fro_norm, parameter in zip(fro_norms, controlled)
    ])
    target_elrs = torch.tensor(
        [
            embedding_elr_multiplier if name == 'transformer.wte.weight' else 1.0
            for name in controlled_names
        ],
        dtype=rms_values.dtype,
        device=device,
    ) * target_elr
    learning_rates = (rms_values * target_elrs).tolist()
    for group, learning_rate in zip(optimizer.param_groups, learning_rates):
        group['lr'] = float(learning_rate)

    actual_elrs = torch.tensor(learning_rates) / rms_values.cpu()
    target_elrs_cpu = target_elrs.cpu()
    max_relative_error = float(((actual_elrs - target_elrs_cpu).abs() / target_elrs_cpu).max())
    assert len(optimizer.param_groups) == 73
    assert len(gamma_optimizer.param_groups) == 1
    assert max_relative_error < 2e-7
    assert abs(float(actual_elrs[-1]) / float(actual_elrs[0]) - 2.0) < 2e-7

    for parameter in controlled:
        parameter.grad = torch.randn_like(parameter)
    gamma.grad = torch.randn_like(gamma)
    optimizer.step()
    gamma_optimizer.step()
    print(f'device={device}')
    print(f'controlled_groups={len(optimizer.param_groups)}')
    print(f'gamma_groups={len(gamma_optimizer.param_groups)}')
    print(f'max_relative_elr_error={max_relative_error:.3e}')
    print('SMOKE_OK')


if __name__ == '__main__':
    main()
