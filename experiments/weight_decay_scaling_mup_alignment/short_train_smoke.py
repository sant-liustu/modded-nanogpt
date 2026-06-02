import argparse
import json
import math
import pathlib

import torch

from init_smoke import load_model_defs


def global_grad_norm(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def run_short_train(args):
    GPT, GPTConfig = load_model_defs()
    torch.manual_seed(args.seed)
    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        init_std=0.02,
        scale_emb=args.alpha_in,
        scale_base_model=args.scale_base_model,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    records = []

    for step in range(args.steps):
        idx = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length))
        targets = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length))
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(idx, targets, return_logits=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = global_grad_norm(model.parameters())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm}")
        optimizer.step()
        records.append(
            {
                "step": step,
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "logits_std": logits.detach().float().std(unbiased=False).item(),
            }
        )

    result = {
        "config": {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()},
        "records": records,
        "all_losses_finite": all(math.isfinite(record["loss"]) for record in records),
        "all_grad_norms_finite": all(math.isfinite(record["grad_norm"]) for record in records),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-in", type=float, default=10.0)
    parser.add_argument("--scale-base-model", type=int, default=768)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    result = run_short_train(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
