"""Build reproducible inputs for the hard-norm ELR-govern stress tests.

This creates four *fixed* random assignments of one norm trajectory per
controlled matrix and the shared single-ELR WSD target.  The assignments are
sampled once with their recorded seeds; training never resamples them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("experiments/norm_control_schedule_collapse")
HARD_NORM_SCHEDULES = (
    "constant",
    "linear_up",
    "linear_down",
    "cosine_cycle",
)
ASSIGNMENT_SPECS = (
    ("singleelr_A", 20260901),
    ("singleelr_B", 20260902),
    ("pertensor_A", 20260903),
    ("pertensor_B", 20260904),
)


def controlled_tensor_names(n_layer: int) -> list[str]:
    names: list[str] = []
    for layer in range(n_layer):
        prefix = f"transformer.h.{layer}"
        for suffix in (
            "attn.c_q.weight",
            "attn.c_k.weight",
            "attn.c_v.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ):
            names.append(f"{prefix}.{suffix}")
    names.append("transformer.wte.weight")
    return sorted(names)


def wsd_ratio(t: int, num_iterations: int, warmup_iters: int, warmdown_iters: int) -> float:
    if t < warmup_iters:
        return (t + 1) / warmup_iters
    if t < num_iterations - warmdown_iters:
        return 1.0
    return (num_iterations - t) / warmdown_iters


def make_assignment(names: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    assignment = {name: rng.choice(HARD_NORM_SCHEDULES) for name in names}
    counts = Counter(assignment.values())
    if set(counts) != set(HARD_NORM_SCHEDULES):
        raise RuntimeError(
            f"seed {seed} did not draw every schedule; choose another reproducible seed"
        )
    return assignment


def write_assignment_config(output_dir: Path, label: str, seed: int, names: list[str]) -> dict[str, object]:
    assignment = make_assignment(names, seed)
    payload: dict[str, object] = {
        "enabled": True,
        "mode": "hard_schedule_from_initial_rms",
        "norm_type": "rms",
        "assignment_seed": seed,
        "selection_method": "random.Random(seed).choice over hard_norm_schedules, sampled once before training",
        "controlled_tensor_count": len(names),
        "hard_norm_schedules": {
            "constant": "q(x) = 1",
            "linear_up": "q(x) = 1 + x; q(1) = 2",
            "linear_down": "q(x) = 1 - 0.5 x; q(1) = 0.5",
            "cosine_cycle": "q(x) = 1 + 0.5 sin(2 pi x); exactly one complete cycle on x in [0, 1]",
        },
        "coordinate": "initial state uses x=0; after update s, project to x=s/num_iterations",
        "eps": 1e-12,
        "log_every": 10,
        "assignments": assignment,
        "assignment_counts": dict(sorted(Counter(assignment.values()).items())),
    }
    path = output_dir / f"hardnorm_assignment_{label}_seed{seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "label": label,
        "seed": seed,
        "path": str(path),
        "assignment_counts": payload["assignment_counts"],
    }


def write_single_elr_target(
    output_dir: Path,
    names: list[str],
    num_iterations: int,
    warmup_iters: int,
    warmdown_iters: int,
    peak_elr: float,
) -> dict[str, object]:
    path = output_dir / "rmselr_single_wsd_peak005_B0128_20400.jsonl.gz"
    digest = hashlib.sha256()
    with path.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed_file:
            for update_step in range(1, num_iterations + 1):
                target_elr = peak_elr * wsd_ratio(
                    update_step - 1,
                    num_iterations,
                    warmup_iters,
                    warmdown_iters,
                )
                record = {
                    "update_step": update_step,
                    "norm_type": "rms",
                    "norm_scope": "per-tensor RMS; lr_i = target_elr_i * RMS(W_i)",
                    "target_lr_over_tensor_rms": {
                        name: target_elr for name in names
                    },
                }
                line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                encoded = line.encode("utf-8")
                compressed_file.write(encoded)
                digest.update(encoded)
    return {
        "path": str(path),
        "rows": num_iterations,
        "tensor_count": len(names),
        "peak_elr": peak_elr,
        "schedule": "WSD with the same value for every controlled tensor",
        "content_sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-iterations", type=int, default=20400)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--warmdown-iters", type=int, default=5800)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--peak-elr", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_layer <= 0:
        raise ValueError("n_layer must be positive")
    if not (0 < args.warmup_iters < args.num_iterations):
        raise ValueError("warmup_iters must lie strictly between 0 and num_iterations")
    if not (0 < args.warmdown_iters < args.num_iterations):
        raise ValueError("warmdown_iters must lie strictly between 0 and num_iterations")
    if args.peak_elr <= 0:
        raise ValueError("peak_elr must be positive")

    names = controlled_tensor_names(args.n_layer)
    expected_count = 6 * args.n_layer + 1
    if len(names) != expected_count:
        raise RuntimeError(f"expected {expected_count} controlled tensors, got {len(names)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = [
        write_assignment_config(args.output_dir, label, seed, names)
        for label, seed in ASSIGNMENT_SPECS
    ]
    target = write_single_elr_target(
        args.output_dir,
        names,
        args.num_iterations,
        args.warmup_iters,
        args.warmdown_iters,
        args.peak_elr,
    )
    print(json.dumps({"assignments": assignments, "single_elr_target": target}, indent=2))


if __name__ == "__main__":
    main()
