#!/usr/bin/env python
"""Compute window-level coherent signal/noise from checkpoints and update norms.

The analyzer estimates how much per-step update energy is coherent across a
checkpoint window. For a window with K optimizer updates:

    D = theta_end - theta_start
    Q = sum_i ||u_i||^2

It reports:

    signal_energy = max(0, (||D||^2 - Q) / (K * (K - 1)))
    total_energy = Q / K
    noise_energy = max(0, total_energy - signal_energy)

This script is intentionally standalone and conservative about units. Raw
checkpoint differences are parameter displacements, so Q must also be parameter
displacement energy. Existing lr-free AdamW direction norms are only converted
when weight_decay is exactly zero, where ||delta|| = lr * ||adamw_update||.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ACTUAL_UPDATE_FIELDS = (
    "actual_update_fro_norm",
    "applied_update_fro_norm",
    "parameter_update_fro_norm",
    "raw_update_fro_norm",
    "delta_fro_norm",
)


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    step: int


@dataclass
class EnergyRecord:
    q: float = 0.0
    rows: int = 0
    source: str | None = None
    skipped_rows: int = 0
    skipped_reasons: dict[str, int] | None = None

    def add_skip(self, reason: str) -> None:
        self.skipped_rows += 1
        if self.skipped_reasons is None:
            self.skipped_reasons = {}
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze D/Q window coherence from saved checkpoints and update norm history."
    )
    parser.add_argument("--run-dir", type=Path, help="Run directory containing checkpoints and norm history.")
    parser.add_argument("--output-dir", type=Path, help="Directory for JSONL/CSV summaries.")
    parser.add_argument("--checkpoint-glob", default="state_step*.pt", help="Checkpoint glob under run-dir.")
    parser.add_argument(
        "--start-checkpoint",
        type=Path,
        help="Explicit start checkpoint for a single window. Requires --end-checkpoint.",
    )
    parser.add_argument(
        "--end-checkpoint",
        type=Path,
        help="Explicit end checkpoint for a single window. Requires --start-checkpoint.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        help="Step for --start-checkpoint when it cannot be parsed from the file name.",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        help="Step for --end-checkpoint when it cannot be parsed from the file name.",
    )
    parser.add_argument(
        "--update-history",
        type=Path,
        help="Update norm JSONL path. Defaults to run-dir/adamw_update_norm_history.jsonl.",
    )
    parser.add_argument(
        "--checkpoint-key",
        default="model",
        help="Key containing the state_dict inside torch checkpoints.",
    )
    parser.add_argument(
        "--min-window-updates",
        type=int,
        default=2,
        help="Minimum K required for signal/noise estimation.",
    )
    parser.add_argument(
        "--strict-complete-q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require one usable update norm row per optimizer step and tensor in a window.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run formula self-test without loading run data.",
    )
    return parser.parse_args()


def discover_checkpoints(run_dir: Path, pattern: str) -> list[CheckpointInfo]:
    checkpoints: list[CheckpointInfo] = []
    for path in sorted(run_dir.glob(pattern)):
        step = parse_step_from_name(path.name)
        if step is not None:
            checkpoints.append(CheckpointInfo(path=path, step=step))
    return sorted(checkpoints, key=lambda item: item.step)


def checkpoint_info_from_path(path: Path, explicit_step: int | None) -> CheckpointInfo:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    step = explicit_step if explicit_step is not None else parse_step_from_name(resolved.name)
    if step is None:
        raise ValueError(f"cannot parse step from {resolved.name}; provide --start-step/--end-step")
    return CheckpointInfo(path=resolved, step=step)


def parse_step_from_name(name: str) -> int | None:
    match = re.search(r"step(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def load_torch_state_dict(path: Path, checkpoint_key: str) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Loading .pt checkpoints requires torch. Use a torch-enabled Python environment."
        ) from exc

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state = checkpoint[checkpoint_key]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state_dict-like object")
    return state


def tensor_d2(start_tensor: Any, end_tensor: Any) -> float:
    # This function expects torch tensors but avoids importing torch globally.
    diff = end_tensor.detach().float() - start_tensor.detach().float()
    return float(diff.square().sum().item())


def tensor_norm2(tensor: Any) -> float:
    t = tensor.detach().float()
    return float(t.square().sum().item())


def is_float_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "is_floating_point") and bool(value.is_floating_point())


def usable_update_norm(row: dict[str, Any]) -> tuple[float | None, str | None, str | None]:
    for field in ACTUAL_UPDATE_FIELDS:
        if field in row:
            return float(row[field]), field, None

    if "adamw_update_fro_norm" in row:
        weight_decay = float(row.get("weight_decay", 0.0))
        lr = row.get("lr")
        if lr is None:
            return None, None, "adamw_update_missing_lr"
        if abs(weight_decay) > 0.0:
            return None, None, "adamw_update_nonzero_weight_decay"
        return abs(float(lr)) * float(row["adamw_update_fro_norm"]), "lr_scaled_adamw_no_decay", None

    return None, None, "no_usable_update_norm_field"


def load_update_energy_by_step(history_path: Path) -> dict[int, dict[str, EnergyRecord]]:
    by_step: dict[int, dict[str, EnergyRecord]] = defaultdict(lambda: defaultdict(EnergyRecord))
    with history_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "step" not in row or "name" not in row:
                raise ValueError(f"{history_path}:{line_number} missing step or name")
            step = int(row["step"])
            name = str(row["name"])
            record = by_step[step][name]
            norm, source, skip_reason = usable_update_norm(row)
            if norm is None:
                record.add_skip(skip_reason or "unknown")
                continue
            record.q += norm * norm
            record.rows += 1
            if record.source is None:
                record.source = source
            elif record.source != source:
                record.source = "mixed"
    return by_step


def aggregate_q(
    by_step: dict[int, dict[str, EnergyRecord]],
    start_step: int,
    end_step: int,
) -> dict[str, EnergyRecord]:
    out: dict[str, EnergyRecord] = defaultdict(EnergyRecord)
    for step in range(start_step + 1, end_step + 1):
        for name, record in by_step.get(step, {}).items():
            dst = out[name]
            dst.q += record.q
            dst.rows += record.rows
            dst.skipped_rows += record.skipped_rows
            if record.skipped_reasons:
                if dst.skipped_reasons is None:
                    dst.skipped_reasons = {}
                for reason, count in record.skipped_reasons.items():
                    dst.skipped_reasons[reason] = dst.skipped_reasons.get(reason, 0) + count
            if dst.source is None:
                dst.source = record.source
            elif record.source is not None and dst.source != record.source:
                dst.source = "mixed"
    return out


def coherence_metrics(d2: float, q: float, k: int, eps: float = 1e-30) -> dict[str, float | None]:
    if k <= 1 or q < 0.0 or d2 < 0.0:
        return {
            "total_energy_per_step": None,
            "signal_energy_per_step": None,
            "noise_energy_per_step": None,
            "signal_norm_per_step": None,
            "noise_norm_per_step": None,
            "signal_fraction": None,
            "noise_fraction": None,
            "snr_norm": None,
            "displacement_efficiency": None,
            "raw_signal_fraction": None,
        }
    if q == 0.0:
        zero = 0.0
        return {
            "total_energy_per_step": zero,
            "signal_energy_per_step": zero,
            "noise_energy_per_step": zero,
            "signal_norm_per_step": zero,
            "noise_norm_per_step": zero,
            "signal_fraction": None,
            "noise_fraction": None,
            "snr_norm": None,
            "displacement_efficiency": None,
            "raw_signal_fraction": None,
        }

    signal_energy = max(0.0, (d2 - q) / (k * (k - 1)))
    total_energy = q / k
    raw_signal_fraction = signal_energy / total_energy if total_energy > 0 else None
    signal_fraction = None if raw_signal_fraction is None else min(1.0, max(0.0, raw_signal_fraction))
    signal_energy_clipped = total_energy * signal_fraction if signal_fraction is not None else signal_energy
    noise_energy = max(0.0, total_energy - signal_energy_clipped)
    signal_norm = math.sqrt(signal_energy_clipped)
    noise_norm = math.sqrt(noise_energy)
    return {
        "total_energy_per_step": total_energy,
        "signal_energy_per_step": signal_energy_clipped,
        "noise_energy_per_step": noise_energy,
        "signal_norm_per_step": signal_norm,
        "noise_norm_per_step": noise_norm,
        "signal_fraction": signal_fraction,
        "noise_fraction": None if signal_fraction is None else 1.0 - signal_fraction,
        "snr_norm": signal_norm / (noise_norm + eps),
        "displacement_efficiency": math.sqrt(d2 / q),
        "raw_signal_fraction": raw_signal_fraction,
    }


def tensor_groups(name: str) -> list[str]:
    groups = ["full_model"]
    if "wte" in name or "lm_head" in name:
        groups.append("embedding_lm_head")
    if ".attn." in name:
        groups.append("attention")
    if ".mlp." in name:
        groups.append("mlp")
    if "transformer.h." in name:
        groups.append("transformer_blocks")
        match = re.search(r"transformer\.h\.(\d+)\.", name)
        if match:
            groups.append(f"layer_{match.group(1)}")
    return groups


def analyze_window(
    start_ckpt: CheckpointInfo,
    end_ckpt: CheckpointInfo,
    checkpoint_key: str,
    q_by_name: dict[str, EnergyRecord],
    strict_complete_q: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    start_state = load_torch_state_dict(start_ckpt.path, checkpoint_key)
    end_state = load_torch_state_dict(end_ckpt.path, checkpoint_key)
    k = end_ckpt.step - start_ckpt.step
    tensor_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    common_names = sorted(set(start_state.keys()) & set(end_state.keys()))
    for name in common_names:
        start_tensor = start_state[name]
        end_tensor = end_state[name]
        if not is_float_tensor(start_tensor) or not is_float_tensor(end_tensor):
            continue
        if tuple(start_tensor.shape) != tuple(end_tensor.shape):
            warnings.append(f"shape mismatch for {name}; skipped")
            continue
        energy = q_by_name.get(name)
        if energy is None:
            continue
        if strict_complete_q and energy.rows != k:
            warnings.append(f"{name}: usable update rows {energy.rows} != K {k}; skipped")
            continue
        if energy.q <= 0.0 and energy.rows <= 0:
            continue

        d2 = tensor_d2(start_tensor, end_tensor)
        theta_start_norm = math.sqrt(tensor_norm2(start_tensor))
        theta_end_norm = math.sqrt(tensor_norm2(end_tensor))
        row: dict[str, Any] = {
            "row_type": "tensor",
            "name": name,
            "step_start": start_ckpt.step,
            "step_end": end_ckpt.step,
            "K": k,
            "D_norm": math.sqrt(d2),
            "D_norm_sq": d2,
            "Q": energy.q,
            "usable_update_rows": energy.rows,
            "update_norm_source": energy.source,
            "theta_start_norm": theta_start_norm,
            "theta_end_norm": theta_end_norm,
            "relative_signal_speed": None,
            "relative_noise_speed": None,
        }
        row.update(coherence_metrics(d2, energy.q, k))
        if row["signal_norm_per_step"] is not None and theta_start_norm > 0:
            row["relative_signal_speed"] = row["signal_norm_per_step"] / theta_start_norm
        if row["noise_norm_per_step"] is not None and theta_start_norm > 0:
            row["relative_noise_speed"] = row["noise_norm_per_step"] / theta_start_norm
        tensor_rows.append(row)

    group_acc: dict[str, dict[str, Any]] = {}
    for row in tensor_rows:
        for group in tensor_groups(row["name"]):
            acc = group_acc.setdefault(
                group,
                {
                    "row_type": "group",
                    "name": group,
                    "step_start": start_ckpt.step,
                    "step_end": end_ckpt.step,
                    "K": k,
                    "D_norm_sq": 0.0,
                    "Q": 0.0,
                    "tensor_count": 0,
                    "theta_start_norm_sq": 0.0,
                    "theta_end_norm_sq": 0.0,
                },
            )
            acc["D_norm_sq"] += float(row["D_norm_sq"])
            acc["Q"] += float(row["Q"])
            acc["tensor_count"] += 1
            acc["theta_start_norm_sq"] += float(row["theta_start_norm"]) ** 2
            acc["theta_end_norm_sq"] += float(row["theta_end_norm"]) ** 2

    group_rows: list[dict[str, Any]] = []
    for acc in sorted(group_acc.values(), key=lambda item: item["name"]):
        d2 = float(acc.pop("D_norm_sq"))
        theta_start_norm = math.sqrt(float(acc.pop("theta_start_norm_sq")))
        theta_end_norm = math.sqrt(float(acc.pop("theta_end_norm_sq")))
        row = dict(acc)
        row["D_norm"] = math.sqrt(d2)
        row["D_norm_sq"] = d2
        row["theta_start_norm"] = theta_start_norm
        row["theta_end_norm"] = theta_end_norm
        row["relative_signal_speed"] = None
        row["relative_noise_speed"] = None
        row.update(coherence_metrics(d2, float(row["Q"]), k))
        if row["signal_norm_per_step"] is not None and theta_start_norm > 0:
            row["relative_signal_speed"] = row["signal_norm_per_step"] / theta_start_norm
        if row["noise_norm_per_step"] is not None and theta_start_norm > 0:
            row["relative_noise_speed"] = row["noise_norm_per_step"] / theta_start_norm
        group_rows.append(row)

    return tensor_rows, group_rows, warnings


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(args: argparse.Namespace) -> int:
    explicit_window = args.start_checkpoint is not None or args.end_checkpoint is not None
    if explicit_window and (args.start_checkpoint is None or args.end_checkpoint is None):
        raise SystemExit("--start-checkpoint and --end-checkpoint must be provided together")
    if explicit_window:
        start_ckpt = checkpoint_info_from_path(args.start_checkpoint, args.start_step)
        end_ckpt = checkpoint_info_from_path(args.end_checkpoint, args.end_step)
        if end_ckpt.step <= start_ckpt.step:
            raise SystemExit("--end-checkpoint step must be greater than --start-checkpoint step")
        run_dir = args.run_dir.resolve() if args.run_dir is not None else start_ckpt.path.parent
        checkpoints = [start_ckpt, end_ckpt]
    else:
        if args.run_dir is None:
            raise SystemExit("--run-dir is required unless --self-test is used")
        run_dir = args.run_dir.resolve()
        checkpoints = discover_checkpoints(run_dir, args.checkpoint_glob)

    output_dir = (args.output_dir or run_dir / "window_coherence").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = (args.update_history or run_dir / "adamw_update_norm_history.jsonl").resolve()

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "update_history": str(history_path),
        "checkpoint_glob": args.checkpoint_glob,
        "explicit_window": explicit_window,
        "output_dir": str(output_dir),
        "status": "started",
        "warnings": [],
    }

    if not history_path.exists():
        summary["status"] = "missing_update_history"
        summary["warnings"].append(f"missing update history: {history_path}")
        write_summary(output_dir, summary)
        print_status(summary)
        return 2

    summary["checkpoints"] = [{"path": str(item.path), "step": item.step} for item in checkpoints]
    if len(checkpoints) < 2:
        summary["status"] = "no_checkpoint_windows"
        summary["warnings"].append(
            f"found {len(checkpoints)} checkpoint(s); need at least 2 to compute D"
        )
        summary["note"] = (
            "The analyzer can still parse update norms, but D requires two saved checkpoints."
        )
        by_step = load_update_energy_by_step(history_path)
        summary["update_steps"] = sorted(by_step.keys())
        summary["usable_update_steps"] = len(summary["update_steps"])
        write_summary(output_dir, summary)
        print_status(summary)
        return 1

    by_step = load_update_energy_by_step(history_path)
    tensor_rows_all: list[dict[str, Any]] = []
    group_rows_all: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []

    for start_ckpt, end_ckpt in zip(checkpoints, checkpoints[1:]):
        k = end_ckpt.step - start_ckpt.step
        window_summary: dict[str, Any] = {
            "step_start": start_ckpt.step,
            "step_end": end_ckpt.step,
            "K": k,
            "start_checkpoint": str(start_ckpt.path),
            "end_checkpoint": str(end_ckpt.path),
        }
        if k < args.min_window_updates:
            window_summary["status"] = "skipped_short_window"
            windows.append(window_summary)
            continue

        q_by_name = aggregate_q(by_step, start_ckpt.step, end_ckpt.step)
        tensor_rows, group_rows, warnings = analyze_window(
            start_ckpt=start_ckpt,
            end_ckpt=end_ckpt,
            checkpoint_key=args.checkpoint_key,
            q_by_name=q_by_name,
            strict_complete_q=args.strict_complete_q,
        )
        window_summary["status"] = "ok" if group_rows else "no_usable_tensors"
        window_summary["tensor_rows"] = len(tensor_rows)
        window_summary["group_rows"] = len(group_rows)
        window_summary["warnings"] = warnings[:20]
        tensor_rows_all.extend(tensor_rows)
        group_rows_all.extend(group_rows)
        windows.append(window_summary)
        summary["warnings"].extend(warnings[:50])

    summary["windows"] = windows
    summary["tensor_rows"] = len(tensor_rows_all)
    summary["group_rows"] = len(group_rows_all)
    summary["status"] = "ok" if group_rows_all else "no_usable_windows"

    write_jsonl(output_dir / "window_coherence_tensors.jsonl", tensor_rows_all)
    write_jsonl(output_dir / "window_coherence_groups.jsonl", group_rows_all)
    write_csv(output_dir / "window_coherence_groups.csv", group_rows_all)
    write_summary(output_dir, summary)
    print_status(summary)
    return 0 if group_rows_all else 1


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_status(summary: dict[str, Any]) -> None:
    print(f"status: {summary.get('status')}")
    print(f"output_dir: {summary.get('output_dir')}")
    if "tensor_rows" in summary:
        print(f"tensor_rows: {summary['tensor_rows']}")
    if "group_rows" in summary:
        print(f"group_rows: {summary['group_rows']}")
    warnings = summary.get("warnings") or []
    for warning in warnings[:8]:
        print(f"warning: {warning}")
    if len(warnings) > 8:
        print(f"warning: ... {len(warnings) - 8} more")


def run_self_test() -> int:
    # Pure noise random-walk expectation: D^2 ~= Q, signal fraction ~= 0.
    noise = coherence_metrics(d2=10.0, q=10.0, k=10)
    assert noise["signal_fraction"] == 0.0, noise
    assert noise["noise_fraction"] == 1.0, noise

    # Fully coherent case: each update is s, so D^2 = K * Q.
    coherent = coherence_metrics(d2=100.0, q=10.0, k=10)
    assert abs(float(coherent["signal_fraction"]) - 1.0) < 1e-12, coherent
    assert float(coherent["noise_fraction"]) == 0.0, coherent

    # Halfway example, chosen so unclipped fraction is in (0, 1).
    mixed = coherence_metrics(d2=55.0, q=10.0, k=10)
    expected = ((55.0 / 10.0) - 1.0) / 9.0
    assert abs(float(mixed["signal_fraction"]) - expected) < 1e-12, mixed
    print("self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    return run_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
