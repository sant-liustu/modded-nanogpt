"""Build block-only-global-F and tied-WTE ELR targets from the WD=0 baseline.

The reference has 72 non-gamma Transformer block matrices and one tied
embedding / LM-head matrix.  For every pre-update state this builder records:

* ``target_block_lr_over_block_fro_norm``: the scalar global-F target for
  the 72 blocks only;
* ``target_wte_elr``: the reference WTE effective learning rate, computed
  using its actual 2x block learning rate.

The two WTE-scope ablation runners consume the same output.  The ordinary
block-only arm uses only the first target; the rescue arm additionally uses
the second target while keeping WTE outside the block-only denominator.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from pathlib import Path


DEFAULT_SOURCE_RUN = Path(
    r"F:\\自动化\\自动化for people\\实验\\已完成实验数据\\expdata\\modded-nanogpt"
    r"\\rmsnorm_gamma_adamw__B0128__devB064__lr0p0036__blocklr0p0018__wd0"
    r"__warmup1000__warmdown5800__seed0__rep01"
)
DEFAULT_OUTPUT = Path(__file__).with_name(
    "reference_rmsnorm_gamma_adamw_wd0_wsd_rep01_blockonly_globalf_wte_elr.jsonl"
)
EXPECTED_BLOCK_TENSORS = 72


def canonical_name(name: str) -> str:
    return name.removeprefix("_orig_mod.")


def interpolate(series: list[tuple[int, float]], step: int) -> float:
    index = bisect_left(series, (step, -math.inf))
    if index < len(series) and series[index][0] == step:
        return series[index][1]
    if index == 0 or index == len(series):
        raise ValueError(
            f"cannot interpolate step={step} from [{series[0][0]}, {series[-1][0]}]"
        )
    left_step, left_value = series[index - 1]
    right_step, right_value = series[index]
    return left_value + (right_value - left_value) * (step - left_step) / (right_step - left_step)


def wsd_block_lr(
    update_step: int,
    *,
    block_lr: float,
    num_iterations: int,
    warmup_iters: int,
    warmdown_iters: int,
) -> float:
    schedule_index = update_step - 1
    if schedule_index < warmup_iters:
        return block_lr * (schedule_index + 1) / warmup_iters
    if schedule_index < num_iterations - warmdown_iters:
        return block_lr
    return block_lr * (num_iterations - schedule_index) / warmdown_iters


def load_reference_series(
    source_run: Path, num_iterations: int
) -> tuple[dict[str, list[tuple[int, float]]], list[tuple[int, float]], str]:
    history_path = source_run / "tensor_norm_history.jsonl"
    block_fro_series: dict[str, list[tuple[int, float]]] = {}
    wte_rms_series: list[tuple[int, float]] = []
    with history_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if int(record["ndim"]) != 2:
                continue
            name = canonical_name(str(record["name"]))
            step = int(record["step"])
            if name.startswith("transformer.h."):
                fro_norm = float(record["fro_norm"])
                if not math.isfinite(fro_norm) or fro_norm <= 0.0:
                    raise ValueError(f"invalid Frobenius norm for {name} at line {line_number}")
                block_fro_series.setdefault(name, []).append((step, fro_norm))
            elif name == "transformer.wte.weight":
                rms = float(record["rms_norm"])
                if not math.isfinite(rms) or rms <= 0.0:
                    raise ValueError(f"invalid RMS for {name} at line {line_number}")
                wte_rms_series.append((step, rms))

    if len(block_fro_series) != EXPECTED_BLOCK_TENSORS:
        raise ValueError(
            f"expected {EXPECTED_BLOCK_TENSORS} block matrices, found {len(block_fro_series)}"
        )
    if not wte_rms_series:
        raise ValueError("missing tied WTE RMS series")
    for name, values in block_fro_series.items():
        values.sort()
        if values[0][0] != 0 or values[-1][0] != num_iterations:
            raise ValueError(
                f"incomplete Frobenius history for {name}: first={values[0][0]}, last={values[-1][0]}"
            )
    wte_rms_series.sort()
    if wte_rms_series[0][0] != 0 or wte_rms_series[-1][0] != num_iterations:
        raise ValueError(
            "incomplete WTE RMS history: "
            f"first={wte_rms_series[0][0]}, last={wte_rms_series[-1][0]}"
        )
    return block_fro_series, wte_rms_series, str(history_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-iterations", type=int, default=20_400)
    parser.add_argument("--block-lr", type=float, default=0.0018)
    parser.add_argument("--warmup-iters", type=int, default=1_000)
    parser.add_argument("--warmdown-iters", type=int, default=5_800)
    parser.add_argument("--wte-to-block-lr-ratio", type=float, default=2.0)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not (0 < args.warmup_iters < args.num_iterations):
        raise ValueError("warmup_iters must lie strictly inside the training interval")
    if not (0 < args.warmdown_iters < args.num_iterations):
        raise ValueError("warmdown_iters must lie strictly inside the training interval")
    if args.block_lr <= 0.0 or args.wte_to_block_lr_ratio <= 0.0:
        raise ValueError("learning-rate scales must be positive")

    block_fro_series, wte_rms_series, history_path = load_reference_series(
        args.source_run, args.num_iterations
    )
    block_names = sorted(block_fro_series)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for update_step in range(1, args.num_iterations + 1):
            pre_update_step = update_step - 1
            reference_block_lr = wsd_block_lr(
                update_step,
                block_lr=args.block_lr,
                num_iterations=args.num_iterations,
                warmup_iters=args.warmup_iters,
                warmdown_iters=args.warmdown_iters,
            )
            block_fro_sq = sum(
                interpolate(block_fro_series[name], pre_update_step) ** 2
                for name in block_names
            )
            reference_block_fro_norm = math.sqrt(block_fro_sq)
            reference_wte_rms = interpolate(wte_rms_series, pre_update_step)
            reference_wte_lr = args.wte_to_block_lr_ratio * reference_block_lr
            record = {
                "update_step": update_step,
                "pre_update_step": pre_update_step,
                "reference_experiment": "rmsnorm_gamma_adamw__B0128__devB064__lr0p0036__blocklr0p0018__wd0__warmup1000__warmdown5800__seed0__rep01",
                "block_tensor_count": EXPECTED_BLOCK_TENSORS,
                "wte_in_block_global_f_denominator": False,
                "reference_block_lr": reference_block_lr,
                "reference_block_fro_norm": reference_block_fro_norm,
                "target_block_lr_over_block_fro_norm": reference_block_lr / reference_block_fro_norm,
                "reference_wte_lr": reference_wte_lr,
                "reference_wte_rms": reference_wte_rms,
                "target_wte_elr": reference_wte_lr / reference_wte_rms,
                "reference_wte_weight_decay": 0.0,
                "interpolation": "linear from 4-step tensor_norm_history",
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_tensor_norm_history": history_path,
                "rows": args.num_iterations,
                "block_tensor_count": EXPECTED_BLOCK_TENSORS,
                "wte_in_block_global_f_denominator": False,
                "reference_wte_to_block_lr_ratio": args.wte_to_block_lr_ratio,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(parse_args())
