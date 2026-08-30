"""Build a deterministic, gzip-compressed RMS-ELR target file for the AdamW pair.

The target row for update step s uses schedule coordinate t=s-1.  Both curves
therefore have value zero at the post-training endpoint t=num_iterations, while
the final actual update retains a small positive LR.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


DEFAULT_OUTPUT = Path(
    "experiments/norm_control_schedule_collapse/"
    "rmselr_mixed_attncos_mlpwsd_peak005_007_B0128_20400.jsonl.gz"
)


def wsd_ratio(t: int, num_iterations: int, warmup_iters: int, warmdown_iters: int) -> float:
    if t < warmup_iters:
        return (t + 1) / warmup_iters
    if t < num_iterations - warmdown_iters:
        return 1.0
    return (num_iterations - t) / warmdown_iters


def cosine_ratio(t: int, num_iterations: int, warmup_iters: int) -> float:
    if t < warmup_iters:
        return (t + 1) / warmup_iters
    decay_progress = (t - warmup_iters) / (num_iterations - warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def tensor_target_specs(n_layer: int, embedding_peak_elr: float) -> dict[str, tuple[float, str]]:
    specs: dict[str, tuple[float, str]] = {}
    for layer in range(n_layer):
        prefix = f"transformer.h.{layer}"
        for name in ("c_q", "c_k", "c_v"):
            specs[f"{prefix}.attn.{name}.weight"] = (0.05, "cosine")
        specs[f"{prefix}.attn.c_proj.weight"] = (0.07, "cosine")
        specs[f"{prefix}.mlp.c_fc.weight"] = (0.05, "wsd")
        specs[f"{prefix}.mlp.c_proj.weight"] = (0.07, "wsd")
    specs["transformer.wte.weight"] = (embedding_peak_elr, "wsd")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-iterations", type=int, default=20400)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--warmdown-iters", type=int, default=5800)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--embedding-peak-elr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.warmup_iters < args.num_iterations):
        raise ValueError("warmup_iters must lie strictly between 0 and num_iterations")
    if not (0 < args.warmdown_iters < args.num_iterations):
        raise ValueError("warmdown_iters must lie strictly between 0 and num_iterations")
    if args.n_layer <= 0 or args.embedding_peak_elr <= 0:
        raise ValueError("n_layer and embedding_peak_elr must be positive")

    specs = tensor_target_specs(args.n_layer, args.embedding_peak_elr)
    expected_count = 6 * args.n_layer + 1
    if len(specs) != expected_count:
        raise RuntimeError(f"expected {expected_count} targets, got {len(specs)}")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    # mtime=0 makes the gzip artifact byte-reproducible across runs.
    with output.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as gzip_file:
            for update_step in range(1, args.num_iterations + 1):
                t = update_step - 1
                wsd = wsd_ratio(t, args.num_iterations, args.warmup_iters, args.warmdown_iters)
                cosine = cosine_ratio(t, args.num_iterations, args.warmup_iters)
                targets = {
                    name: peak * (cosine if schedule == "cosine" else wsd)
                    for name, (peak, schedule) in sorted(specs.items())
                }
                record = dict(
                    update_step=update_step,
                    norm_type="rms",
                    norm_scope="per-tensor RMS; lr_i = target_elr_i * RMS(W_i)",
                    target_lr_over_tensor_norms=targets,
                )
                line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                encoded = line.encode("utf-8")
                gzip_file.write(encoded)
                digest.update(encoded)

    print(json.dumps(dict(
        output=str(output),
        rows=args.num_iterations,
        tensor_count=len(specs),
        content_sha256=digest.hexdigest(),
        attention_schedule="cosine with linear warmup; endpoint t=num_iterations is zero",
        mlp_schedule="WSD with linear warmup and warmdown; endpoint t=num_iterations is zero",
        qkv_peak_elr=0.05,
        attention_output_peak_elr=0.07,
        mlp_up_peak_elr=0.05,
        mlp_down_peak_elr=0.07,
        tied_embedding_peak_elr=args.embedding_peak_elr,
    ), indent=2))


if __name__ == "__main__":
    main()
