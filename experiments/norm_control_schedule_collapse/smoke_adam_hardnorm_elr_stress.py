"""CUDA smoke test for all four hard-norm ELR-govern stress-test runners."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
VARIANTS = (
    (
        "experiment1_assignmentA",
        "train_gpt2_gamma_adam_hardnorm_singleelr_wsd005_assignmentA_muonhinit_B0128_devB064.py",
        "hardnorm_assignment_singleelr_A_seed20260901.json",
        "rmselr_single_wsd_peak005_B0128_20400.jsonl.gz",
        20260901,
    ),
    (
        "experiment1_assignmentB",
        "train_gpt2_gamma_adam_hardnorm_singleelr_wsd005_assignmentB_muonhinit_B0128_devB064.py",
        "hardnorm_assignment_singleelr_B_seed20260902.json",
        "rmselr_single_wsd_peak005_B0128_20400.jsonl.gz",
        20260902,
    ),
    (
        "experiment2_assignmentA",
        "train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_assignmentA_muonhinit_B0128_devB064.py",
        "hardnorm_assignment_pertensor_A_seed20260903.json",
        "rmselr_mixed_attncos_mlpwsd_peak005_007_B0128_20400.jsonl.gz",
        20260903,
    ),
    (
        "experiment2_assignmentB",
        "train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_assignmentB_muonhinit_B0128_devB064.py",
        "hardnorm_assignment_pertensor_B_seed20260904.json",
        "rmselr_mixed_attncos_mlpwsd_peak005_007_B0128_20400.jsonl.gz",
        20260904,
    ),
)


def write_fineweb_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    path.write_bytes(header.tobytes() + tokens.astype(np.uint16).tobytes())


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise AssertionError(f"expected one {label} replacement, found {count}")
    return source.replace(old, new, 1)


def make_tiny_source(source_path: Path, output_path: Path) -> None:
    source = source_path.read_text(encoding="utf-8")
    replacements = (
        ("batch_size : int = 128", "batch_size : int = 2", "global batch"),
        ("device_batch_size : int = 64", "device_batch_size : int = 2", "device batch"),
        ("sequence_length : int = 1024", "sequence_length : int = 16", "sequence length"),
        ("num_iterations : int = 20400", "num_iterations : int = 4", "update count"),
        ("warmup_iters : int = 1000", "warmup_iters : int = 1", "warmup"),
        ("warmdown_iters : int = 5800", "warmdown_iters : int = 1", "warmdown"),
        ("val_loss_every : int = 500", "val_loss_every : int = 0", "validation cadence"),
        ("val_tokens : int = 10485760", "val_tokens : int = 32", "validation tokens"),
        ("compile_model : int = 1", "compile_model : int = 0", "model compilation"),
        ("tensor_norm_every : int = 4", "tensor_norm_every : int = 1", "tensor logging"),
        ("adamw_update_norm_every : int = 4", "adamw_update_norm_every : int = 1", "update logging"),
        (
            "spectral_norm_estimate_enabled : int = 1",
            "spectral_norm_estimate_enabled : int = 0",
            "spectral logging",
        ),
        ("num_vocab = 50304", "num_vocab = 256", "vocabulary"),
        (
            "GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768)",
            "GPTConfig(vocab_size=num_vocab, n_layer=2, n_head=2, n_embd=32)",
            "model shape",
        ),
        (
            "if len(expected_elr_names) != 73 or 'transformer.wte.weight' not in expected_elr_names:",
            "if len(expected_elr_names) != 13 or 'transformer.wte.weight' not in expected_elr_names:",
            "tiny controlled count",
        ),
        (
            "if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):",
            "if master_process and args.save_every > 0 and step % args.save_every == 0:",
            "checkpoint suppression",
        ),
    )
    for old, new, label in replacements:
        source = replace_once(source, old, new, label)
    output_path.write_text(source, encoding="utf-8")


def tiny_names() -> list[str]:
    names: list[str] = []
    for layer in range(2):
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


def write_tiny_config(path: Path, names: list[str]) -> dict[str, str]:
    schedules = ("constant", "linear_up", "linear_down", "cosine_cycle")
    assignments = {name: schedules[index % len(schedules)] for index, name in enumerate(names)}
    payload = {
        "enabled": True,
        "mode": "hard_schedule_from_initial_rms",
        "norm_type": "rms",
        "assignment_seed": 0,
        "selection_method": "smoke: deterministic cycling through all four schedules",
        "controlled_tensor_count": len(names),
        "hard_norm_schedules": {},
        "coordinate": "initial x=0; after update s, x=s/num_iterations",
        "eps": 1e-12,
        "log_every": 1,
        "assignments": assignments,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return assignments


def write_tiny_targets(path: Path, names: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for update_step in range(1, 5):
            target = 0.01 * update_step
            handle.write(json.dumps({
                "update_step": update_step,
                "norm_type": "rms",
                "target_lr_over_tensor_rms": {name: target for name in names},
            }) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def expected_ratio(schedule: str, update_step: int) -> float:
    x = update_step / 4
    if schedule == "constant":
        return 1.0
    if schedule == "linear_up":
        return 1.0 + x
    if schedule == "linear_down":
        return 1.0 - 0.5 * x
    if schedule == "cosine_cycle":
        return 1.0 + 0.5 * math.sin(2.0 * math.pi * x)
    raise AssertionError(schedule)


def validate_static_inputs(
    source_path: Path,
    config_path: Path,
    target_name: str,
    expected_seed: int,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    if f"norm_control_config : str = 'experiments/norm_control_schedule_collapse/{config_path.name}'" not in source:
        raise AssertionError(f"wrong default assignment config in {source_path.name}")
    if f"per_tensor_elr_file : str = 'experiments/norm_control_schedule_collapse/{target_name}'" not in source:
        raise AssertionError(f"wrong default ELR target in {source_path.name}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["assignment_seed"] != expected_seed or len(config["assignments"]) != 73:
        raise AssertionError(f"invalid formal hard-norm config: {config_path.name}")
    if set(config["assignments"].values()) != {
        "constant", "linear_up", "linear_down", "cosine_cycle"
    }:
        raise AssertionError(f"formal config omits a hard norm schedule: {config_path.name}")


def validate_run(run_dir: Path, names: list[str], assignments: dict[str, str]) -> None:
    elr_rows = read_jsonl(run_dir / "per_tensor_elr_history.jsonl")
    if [row["update_step"] for row in elr_rows] != [1, 2, 3, 4]:
        raise AssertionError("ELR history must cover exactly four updates")
    for row in elr_rows:
        target = 0.01 * row["update_step"]
        if set(row["target_lr_over_tensor_rms"]) != set(names):
            raise AssertionError("ELR history names do not match controlled tensors")
        for name in names:
            actual = row["actual_lr_over_tensor_rms"][name]
            if abs(actual - target) > 1e-12:
                raise AssertionError(f"RMS ELR mismatch for {name}: {actual} vs {target}")

    control_rows = read_jsonl(run_dir / "norm_control_history.jsonl")
    initial = [row for row in control_rows if row["event"] == "initial"]
    if len(initial) != len(names):
        raise AssertionError("each controlled tensor must capture its step-0 RMS")
    initial_by_name = {row["name"]: row for row in initial}
    for name, row in initial_by_name.items():
        if row["captured"] is not True or row["projected"] is not False:
            raise AssertionError("initial control must capture without projection")
        if row["schedule"] != assignments[name] or row["schedule_coordinate"] != 0.0:
            raise AssertionError("initial schedule assignment is inconsistent")
        if abs(row["pre_control_rms"] - row["post_control_rms"]) > 1e-12:
            raise AssertionError("step-0 capture changed a parameter")

    projected = [row for row in control_rows if row["event"] == "post_step"]
    if len(projected) != 4 * len(names):
        raise AssertionError("missing post-step hard norm projections")
    for row in projected:
        update_step = row["step"]
        name = row["name"]
        ratio = expected_ratio(assignments[name], update_step)
        expected_target = initial_by_name[name]["initial_rms"] * ratio
        if row["captured"] or not row["projected"]:
            raise AssertionError("post-step event must be a projection")
        if abs(row["schedule_coordinate"] - update_step / 4) > 1e-12:
            raise AssertionError("hard norm schedule coordinate is incorrect")
        if abs(row["schedule_ratio"] - ratio) > 1e-12:
            raise AssertionError("hard norm schedule ratio is incorrect")
        if abs(row["target_rms"] - expected_target) > 1e-10:
            raise AssertionError("hard norm target is inconsistent with captured RMS")
        if row["relative_error"] > 2e-5:
            raise AssertionError(f"hard norm projection error is too high: {row}")

    by_schedule = {
        schedule: [
            row for row in projected if row["schedule"] == schedule and row["step"] == 4
        ]
        for schedule in set(assignments.values())
    }
    if not all(by_schedule.values()):
        raise AssertionError("smoke did not exercise all four schedule endpoints")
    if any(abs(row["schedule_ratio"] - 2.0) > 1e-12 for row in by_schedule["linear_up"]):
        raise AssertionError("linear_up must end at 2x")
    if any(abs(row["schedule_ratio"] - 0.5) > 1e-12 for row in by_schedule["linear_down"]):
        raise AssertionError("linear_down must end at 0.5x")
    if any(abs(row["schedule_ratio"] - 1.0) > 1e-12 for row in by_schedule["cosine_cycle"]):
        raise AssertionError("one-cycle cosine must return to 1x at the end")
    cosine_ratios = [
        row["schedule_ratio"] for row in projected if row["schedule"] == "cosine_cycle"
    ]
    if not (any(abs(value - 1.5) < 1e-12 for value in cosine_ratios)
            and any(abs(value - 0.5) < 1e-12 for value in cosine_ratios)):
        raise AssertionError("cosine_cycle must attain one 1.5x peak and one 0.5x trough")


def main() -> None:
    names = tiny_names()
    if len(names) != 13:
        raise AssertionError(f"expected 13 tiny controlled tensors, got {len(names)}")
    for _, script_name, config_name, target_name, seed in VARIANTS:
        script_path = HERE / script_name
        config_path = HERE / config_name
        if not script_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"formal input missing for {script_name}")
        validate_static_inputs(script_path, config_path, target_name, seed)

    with tempfile.TemporaryDirectory(prefix="adam_hardnorm_elr_stress_smoke_") as temp_string:
        temp_dir = Path(temp_string)
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        tokens = np.arange(512, dtype=np.uint16) % 256
        write_fineweb_shard(data_dir / "fineweb_train_000000.bin", tokens)
        write_fineweb_shard(data_dir / "fineweb_val_000000.bin", tokens[::-1].copy())

        for label, script_name, _, _, _ in VARIANTS:
            run_dir = temp_dir / label
            run_dir.mkdir()
            tiny_script = run_dir / "train_smoke.py"
            make_tiny_source(HERE / script_name, tiny_script)
            assignments = write_tiny_config(run_dir / "hardnorm.json", names)
            targets = run_dir / "targets.jsonl"
            write_tiny_targets(targets, names)
            command = [
                sys.executable,
                str(tiny_script),
                f"--input_bin={data_dir / 'fineweb_train_*.bin'}",
                f"--input_val_bin={data_dir / 'fineweb_val_*.bin'}",
                f"--norm_control_config={run_dir / 'hardnorm.json'}",
                f"--per_tensor_elr_file={targets}",
            ]
            environment = os.environ.copy()
            environment.setdefault("PYTHONUTF8", "1")
            subprocess.run(command, cwd=run_dir, env=environment, check=True)
            log_dirs = [path for path in (run_dir / "logs").iterdir() if path.is_dir()]
            if len(log_dirs) != 1:
                raise AssertionError(f"expected one smoke log directory for {label}, got {log_dirs}")
            validate_run(log_dirs[0], names, assignments)
            print(f"{label}: PASS")

    print("All four Adam hard-norm ELR-govern CUDA smokes: PASS")


if __name__ == "__main__":
    main()
