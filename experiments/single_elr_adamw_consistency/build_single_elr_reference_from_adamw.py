"""Build the 20,400-step block-mean ELR replay target from the completed AdamW run."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from pathlib import Path


DEFAULT_SOURCE_RUN = Path(
    r"F:\自动化\自动化for people\实验\已完成实验数据\expdata\modded-nanogpt"
    r"\rmsnorm_gamma_adamw__B0128__devB064__lr0p0036__blocklr0p0018__wd0p1"
    r"__warmup1000__warmdown5800__seed0__rep01"
)
DEFAULT_OUTPUT = Path(
    "experiments/single_elr_adamw_consistency/single_elr_reference.jsonl"
)


def canonical_name(name: str) -> str:
    return name.removeprefix("_orig_mod.")


def load_block_rms_series(source_run: Path) -> dict[str, list[tuple[int, float]]]:
    history_path = source_run / "tensor_norm_history.jsonl"
    if not history_path.exists():
        raise FileNotFoundError(history_path)

    series: dict[str, list[tuple[int, float]]] = {}
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["ndim"]) != 2:
                continue
            name = canonical_name(record["name"])
            if name == "transformer.wte.weight":
                continue
            if not name.startswith("transformer.h."):
                raise ValueError(f"unexpected non-embedding 2-D tensor: {name}")
            rms = float(record["rms_norm"])
            if not math.isfinite(rms) or rms <= 0.0:
                raise ValueError(f"invalid RMS for {name} at step {record['step']}: {rms}")
            series.setdefault(name, []).append((int(record["step"]), rms))

    if len(series) != 72:
        raise ValueError(f"expected 72 non-embedding block matrices, found {len(series)}")
    for name, values in series.items():
        values.sort()
        if values[0][0] != 0 or values[-1][0] != 20400:
            raise ValueError(
                f"incomplete RMS history for {name}: first={values[0][0]}, last={values[-1][0]}"
            )
    return series


def interpolate(series: list[tuple[int, float]], step: int) -> float:
    index = bisect_left(series, (step, -math.inf))
    if index < len(series) and series[index][0] == step:
        return series[index][1]
    if index == 0 or index == len(series):
        raise ValueError(f"cannot interpolate step {step}")
    left_step, left_value = series[index - 1]
    right_step, right_value = series[index]
    fraction = (step - left_step) / (right_step - left_step)
    return left_value + fraction * (right_value - left_value)


def scheduled_lr_for_update(
    update_step: int,
    *,
    base_lr: float,
    num_updates: int,
    warmup_updates: int,
    warmdown_updates: int,
) -> float:
    schedule_index = update_step - 1
    if schedule_index < warmup_updates:
        multiplier = (schedule_index + 1) / warmup_updates
    elif schedule_index < num_updates - warmdown_updates:
        multiplier = 1.0
    else:
        multiplier = (num_updates - schedule_index) / warmdown_updates
    return base_lr * multiplier


def build(args: argparse.Namespace) -> None:
    series_by_name = load_block_rms_series(args.source_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    minimum = math.inf
    maximum = 0.0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for update_step in range(1, args.num_updates + 1):
            pre_update_step = update_step - 1
            block_lr = scheduled_lr_for_update(
                update_step,
                base_lr=args.block_lr,
                num_updates=args.num_updates,
                warmup_updates=args.warmup_updates,
                warmdown_updates=args.warmdown_updates,
            )
            tensor_elrs = [
                block_lr / interpolate(series, pre_update_step)
                for series in series_by_name.values()
            ]
            target_elr = sum(tensor_elrs) / len(tensor_elrs)
            if not math.isfinite(target_elr) or target_elr <= 0.0:
                raise ValueError(f"invalid target ELR at update {update_step}: {target_elr}")
            minimum = min(minimum, target_elr)
            maximum = max(maximum, target_elr)
            handle.write(json.dumps({"update_step": update_step, "target_elr": target_elr}) + "\n")

    print(f"source_run={args.source_run}")
    print(f"output={args.output}")
    print(f"block_tensor_count={len(series_by_name)}")
    print(f"updates={args.num_updates}")
    print(f"target_elr_min={minimum:.12g}")
    print(f"target_elr_max={maximum:.12g}")
    print("embedding_elr_policy=2 * target_elr (applied by trainer)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-updates", type=int, default=20400)
    parser.add_argument("--block-lr", type=float, default=0.0018)
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--warmdown-updates", type=int, default=5800)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
