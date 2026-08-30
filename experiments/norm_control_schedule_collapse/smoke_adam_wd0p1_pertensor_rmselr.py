"""Two-update CUDA smoke for the AdamW WD=0.1 per-tensor RMS-ELR arm."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
FORMAL_SCRIPT = HERE / (
    "train_gpt2_gamma_adam_wd0p1_pertensor_rmselr_"
    "muonhinit_B0128_devB064.py"
)
FORMAL_CONFIG = HERE / "fixed_initial_norm_all_matrices_start0.json"


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
        ("num_iterations : int = 20400", "num_iterations : int = 2", "update count"),
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
            "if len(controlled_names) != 73 or len(set(controlled_names)) != 73:",
            "if len(controlled_names) != 13 or len(set(controlled_names)) != 13:",
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
    config = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8-sig"))
    names = [
        target["pattern"]
        for target in config["targets"]
        if target["pattern"].startswith("transformer.h.0.")
        or target["pattern"].startswith("transformer.h.1.")
        or target["pattern"] == "transformer.wte.weight"
    ]
    if len(names) != 13 or len(set(names)) != 13:
        raise AssertionError(f"expected 13 tiny controlled tensors, got {len(names)}")
    return names


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_run(run_dir: Path, names: list[str]) -> None:
    history = read_jsonl(run_dir / "per_tensor_rms_elr_history.jsonl")
    if [row["update_step"] for row in history] != [1, 2]:
        raise AssertionError("RMS-ELR history must cover update steps 1 and 2")
    expected_targets = {1: 0.01, 2: 0.02}
    for row in history:
        if row["elr_definition"] != "learning_rate / parameter_rms":
            raise AssertionError("controller did not use parameter RMS")
        if row["weight_decay"] != 0.1 or row["norm_projection"]:
            raise AssertionError("WD arm must use WD=0.1 and no projection")
        if set(row["target_lr_over_tensor_norms"]) != set(names):
            raise AssertionError("RMS-ELR history names do not match controlled tensors")
        target = expected_targets[row["update_step"]]
        for value in row["target_lr_over_tensor_norms"].values():
            if value != target:
                raise AssertionError(f"unexpected target ELR {value} vs {target}")
        if row["max_relative_error"] > 1e-12:
            raise AssertionError(f"RMS-ELR mismatch: {row['max_relative_error']}")

    if (run_dir / "norm_control_history.jsonl").exists():
        raise AssertionError("WD arm unexpectedly wrote norm-control projection history")
    tensor_rows = read_jsonl(run_dir / "tensor_norm_history.jsonl")
    step0 = {row["name"]: row for row in tensor_rows if row["step"] == 0}
    step2 = {row["name"]: row for row in tensor_rows if row["step"] == 2}
    changed = 0
    for name in names:
        if name != "transformer.wte.weight":
            shape = step0[name]["shape"]
            expected_rms = 1.0 / np.sqrt(shape[1])
            relative_error = abs(step0[name]["rms_norm"] - expected_rms) / expected_rms
            if relative_error > 0.12:
                raise AssertionError(f"MuonH initialization RMS mismatch for {name}")
        if step0[name]["rms_norm"] != step2[name]["rms_norm"]:
            changed += 1
    if changed == 0:
        raise AssertionError("all controlled norms stayed fixed despite projection being disabled")


def main() -> None:
    if not FORMAL_SCRIPT.exists() or not FORMAL_CONFIG.exists():
        raise FileNotFoundError("formal script/config missing")
    with tempfile.TemporaryDirectory(prefix="adam_wd0p1_rmselr_smoke_") as temp_string:
        temp_dir = Path(temp_string)
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        tokens = np.arange(256, dtype=np.uint16) % 256
        write_fineweb_shard(data_dir / "fineweb_train_000000.bin", tokens)
        write_fineweb_shard(data_dir / "fineweb_val_000000.bin", tokens[::-1].copy())

        names = tiny_names()
        elr_path = temp_dir / "rms_elr.jsonl"
        with elr_path.open("w", encoding="utf-8", newline="\n") as handle:
            for update_step, target in ((1, 0.01), (2, 0.02)):
                handle.write(json.dumps({
                    "update_step": update_step,
                    "norm_type": "rms",
                    "target_lr_over_tensor_norms": {name: target for name in names},
                }) + "\n")

        tiny_script = temp_dir / "train_smoke.py"
        make_tiny_source(FORMAL_SCRIPT, tiny_script)
        command = [
            sys.executable,
            str(tiny_script),
            f"--input_bin={data_dir / 'fineweb_train_*.bin'}",
            f"--input_val_bin={data_dir / 'fineweb_val_*.bin'}",
            f"--per_tensor_elr_file={elr_path}",
            "--per_tensor_elr_log_every=1",
            "--per_tensor_elr_detailed_log_every=1",
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTHONUTF8", "1")
        subprocess.run(command, cwd=temp_dir, env=environment, check=True)
        run_dirs = [path for path in (temp_dir / "logs").iterdir() if path.is_dir()]
        if len(run_dirs) != 1:
            raise AssertionError(f"expected one run directory, got {run_dirs}")
        validate_run(run_dirs[0], names)
    print("AdamW WD0.1 per-tensor RMS-ELR CUDA smoke: PASS")


if __name__ == "__main__":
    main()
