import argparse
import json
import pathlib

import torch


def load_model_defs():
    root = pathlib.Path(__file__).resolve().parents[2]
    train_py = root / "train_gpt2.py"
    source = train_py.read_text(encoding="utf-8")
    prefix = source.split("# -----------------------------------------------------------------------------\n# Our own simple Distributed Data Loader", 1)[0]
    namespace = {"__file__": str(train_py), "__name__": "mup_parameterization_defs"}
    exec(compile(prefix, str(train_py), "exec"), namespace)
    return namespace["GPT"], namespace["GPTConfig"]


def rms(x):
    return x.detach().float().pow(2).mean().sqrt().item()


def summarize_param(name, tensor):
    data = tensor.detach().float()
    return {
        "name": name,
        "shape": list(data.shape),
        "mean": data.mean().item(),
        "std": data.std(unbiased=False).item(),
        "rms": rms(data),
    }


def run_smoke(scale_emb, scale_base_model, n_embd, n_head, n_layer, sequence_length, batch_size, vocab_size):
    GPT, GPTConfig = load_model_defs()
    torch.manual_seed(1234)
    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        init_std=0.02,
        scale_emb=scale_emb,
        scale_base_model=scale_base_model,
    )
    model = GPT(config)
    model.eval()

    idx = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))
    capture = {}
    handles = []

    def save_output(key):
        def hook(_module, _inputs, output):
            capture.setdefault(key, []).append(rms(output))
        return hook

    for layer, block in enumerate(model.transformer.h):
        handles.append(block.attn.register_forward_hook(save_output(f"layer_{layer}.attn_branch_rms")))
        handles.append(block.mlp.register_forward_hook(save_output(f"layer_{layer}.mlp_branch_rms")))
        handles.append(block.register_forward_hook(save_output(f"layer_{layer}.residual_after_block_rms")))

    with torch.no_grad():
        embedding_raw = model.transformer.wte(idx)
        embedding_scaled = embedding_raw * model.config.scale_emb
        logits, loss = model(idx, targets, return_logits=True)

    for handle in handles:
        handle.remove()

    param_names = [
        "transformer.wte.weight",
        "transformer.h.0.attn.c_q.weight",
        "transformer.h.0.attn.c_k.weight",
        "transformer.h.0.attn.c_v.weight",
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.0.mlp.c_proj.weight",
    ]
    param_dict = dict(model.named_parameters())
    result = {
        "config": {
            "alpha_in": scale_emb,
            "scale_emb": scale_emb,
            "scale_base_model": scale_base_model,
            "width_multiplier": model.width_multiplier,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "init_std": model.config.init_std,
            "attention_scale": "one_over_d_head",
            "expected_hidden_std": 0.02 / (n_embd / scale_base_model) ** 0.5,
            "expected_c_proj_std": 0.02 / (2 * n_layer * n_embd / scale_base_model) ** 0.5,
        },
        "activation": {
            "embedding_raw_rms": rms(embedding_raw),
            "embedding_scaled_rms": rms(embedding_scaled),
            "logits_std": logits.detach().float().std(unbiased=False).item(),
            "logits_rms": rms(logits),
            "initial_loss": loss.item(),
            **{key: values for key, values in capture.items()},
        },
        "parameters": [
            summarize_param(name, param_dict[name])
            for name in param_names
            if name in param_dict
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-in", type=float, nargs="+", default=[1.0, 10.0, 15.0])
    parser.add_argument("--scale-base-model", type=int, default=768)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    results = [
        run_smoke(
            scale_emb=alpha,
            scale_base_model=args.scale_base_model,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
        )
        for alpha in args.alpha_in
    ]

    text = json.dumps(results, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
