"""Build block-mean and tied-embedding ELR targets from the WD=0.1 WSD AdamW baseline."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from pathlib import Path


SOURCE_RUN = Path(
    r"F:\自动化\自动化for people\实验\已完成实验数据\expdata\modded-nanogpt"
    r"\rmsnorm_gamma_adamw__B0128__devB064__lr0p0036__blocklr0p0018__wd0p1"
    r"__warmup1000__warmdown5800__seed0__rep01"
)
OUTPUT = Path(__file__).with_name("global_mean_elr_reference.jsonl")
NUM_UPDATES = 20_400
BLOCK_LR = 0.0018
WARMUP = 1000
WARMDOWN = 5800


def interpolate(series: list[tuple[int, float]], step: int) -> float:
    index = bisect_left(series, (step, -math.inf))
    if index < len(series) and series[index][0] == step:
        return series[index][1]
    left_step, left_value = series[index - 1]
    right_step, right_value = series[index]
    return left_value + (right_value - left_value) * (step - left_step) / (right_step - left_step)


def scheduled_block_lr(update_step: int) -> float:
    schedule_index = update_step - 1
    if schedule_index < WARMUP:
        return BLOCK_LR * (schedule_index + 1) / WARMUP
    if schedule_index < NUM_UPDATES - WARMDOWN:
        return BLOCK_LR
    return BLOCK_LR * (NUM_UPDATES - schedule_index) / WARMDOWN


def main() -> None:
    series: dict[str, list[tuple[int, float]]] = {}
    for line in (SOURCE_RUN / "tensor_norm_history.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["ndim"]) != 2:
            continue
        name = record["name"].removeprefix("_orig_mod.")
        if name == "transformer.wte.weight" or name.startswith("transformer.h."):
            series.setdefault(name, []).append((int(record["step"]), float(record["rms_norm"])))
    if len(series) != 73 or "transformer.wte.weight" not in series:
        raise ValueError(f"expected 72 block matrices plus tied embedding, found {len(series)} tensors")
    for values in series.values():
        values.sort()
        if values[0][0] != 0 or values[-1][0] != NUM_UPDATES:
            raise ValueError("incomplete baseline RMS history")
    block_series = [values for name, values in series.items() if name != "transformer.wte.weight"]
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for update_step in range(1, NUM_UPDATES + 1):
            block_lr = scheduled_block_lr(update_step)
            reference_step = update_step - 1
            block_mean_elr = sum(block_lr / interpolate(values, reference_step) for values in block_series) / len(block_series)
            embedding_elr = 2.0 * block_lr / interpolate(series["transformer.wte.weight"], reference_step)
            handle.write(json.dumps({"update_step": update_step, "target_block_mean_elr": block_mean_elr, "target_embedding_elr": embedding_elr}) + "\n")
    print(f"wrote {NUM_UPDATES} targets to {OUTPUT}")


if __name__ == "__main__":
    main()
