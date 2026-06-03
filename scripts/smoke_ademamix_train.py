from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parents[1]
LOCAL_DEBUG_DIR = REPO_DIR / "data" / "local_debug" / "ademamix_smoke"
TEMP_SCRIPT = LOCAL_DEBUG_DIR / "train_ademamix_smoke.py"


FIELD_TYPES = {
    "input_bin": "str",
    "input_val_bin": "str",
    "batch_size": "int",
    "device_batch_size": "int",
    "sequence_length": "int",
    "num_iterations": "int",
    "embed_learning_rate": "float",
    "warmup_iters": "int",
    "warmdown_iters": "int",
    "weight_decay": "float",
    "optimizer2_type": "str",
    "adema_beta1": "float",
    "adema_beta2": "float",
    "adema_beta3": "float",
    "adema_alpha": "float",
    "adema_beta3_warmup": "int",
    "adema_alpha_warmup": "int",
    "adema_eps": "float",
    "val_loss_every": "int",
    "val_tokens": "int",
    "save_every": "int",
    "compile_model": "int",
    "tensor_norm_every": "int",
    "adamw_update_norm_every": "int",
    "activation_probe_every": "int",
    "spectral_norm_estimate_enabled": "int",
}


def format_value(value: object) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def replace_field(lines: list[str], name: str, value: object) -> None:
    pattern = re.compile(rf"^(\s*){re.escape(name)}\s*:\s*([^=]+?)\s*=\s*(.*?)(\s+#.*)?$")
    matches = [idx for idx, line in enumerate(lines) if pattern.match(line)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Hyperparameters line for {name}, found {len(matches)}")
    idx = matches[0]
    match = pattern.match(lines[idx])
    assert match is not None
    indent, existing_type, _, comment = match.groups()
    expected_type = FIELD_TYPES[name]
    if existing_type.strip() != expected_type:
        raise RuntimeError(f"{name} type changed: expected {expected_type}, found {existing_type.strip()}")
    lines[idx] = f"{indent}{name} : {expected_type} = {format_value(value)}{comment or ''}\n"


def replace_model_config(lines: list[str], n_layer: int, n_head: int, n_embd: int) -> None:
    old = "model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))\n"
    new = f"model = GPT(GPTConfig(vocab_size=num_vocab, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}))\n"
    matches = [idx for idx, line in enumerate(lines) if line == old]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one model config line, found {len(matches)}")
    lines[matches[0]] = new


def write_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def prepare_local_data(num_tokens: int, vocab_size: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(123)
    train_tokens = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    val_tokens = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    train_path = LOCAL_DEBUG_DIR / "fineweb_train_000000.bin"
    val_path = LOCAL_DEBUG_DIR / "fineweb_val_000000.bin"
    write_shard(train_path, train_tokens)
    write_shard(val_path, val_tokens)
    return train_path, val_path


def render_temp_script(args: argparse.Namespace, train_pattern: str, val_pattern: str) -> None:
    template = (REPO_DIR / "train_gpt2.py").read_text(encoding="utf-8").splitlines(keepends=True)
    replacements = {
        "input_bin": train_pattern,
        "input_val_bin": val_pattern,
        "batch_size": args.batch_size,
        "device_batch_size": args.device_batch_size,
        "sequence_length": args.sequence_length,
        "num_iterations": args.num_iterations,
        "embed_learning_rate": args.embed_learning_rate,
        "warmup_iters": args.warmup_iters,
        "warmdown_iters": args.warmdown_iters,
        "weight_decay": args.weight_decay,
        "optimizer2_type": "ademamix",
        "adema_beta1": args.adema_beta1,
        "adema_beta2": args.adema_beta2,
        "adema_beta3": args.adema_beta3,
        "adema_alpha": args.adema_alpha,
        "adema_beta3_warmup": args.adema_beta3_warmup,
        "adema_alpha_warmup": args.adema_alpha_warmup,
        "adema_eps": args.adema_eps,
        "val_loss_every": args.val_loss_every,
        "val_tokens": args.val_tokens,
        "save_every": args.save_every,
        "compile_model": 0,
        "tensor_norm_every": 1,
        "adamw_update_norm_every": 1,
        "activation_probe_every": 1,
        "spectral_norm_estimate_enabled": 0,
    }
    for name, value in replacements.items():
        replace_field(template, name, value)
    replace_model_config(template, args.n_layer, args.n_head, args.n_embd)
    TEMP_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    TEMP_SCRIPT.write_text("".join(template), encoding="utf-8", newline="\n")
    shutil.copy2(REPO_DIR / "ademamix.py", TEMP_SCRIPT.parent / "ademamix.py")


def newest_log_dir(before: set[Path]) -> Path:
    candidates = [path for path in (REPO_DIR / "logs").glob("*") if path.is_dir() and path not in before]
    if not candidates:
        raise RuntimeError("training smoke did not create a new log directory")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def inspect_outputs(log_dir: Path) -> dict[str, object]:
    update_history = log_dir / "adamw_update_norm_history.jsonl"
    tensor_history = log_dir / "tensor_norm_history.jsonl"
    activation_summary = log_dir / "activation_probe_summary.jsonl"
    checkpoints = sorted(log_dir.glob("state_step*.pt"))
    if not update_history.exists():
        raise RuntimeError(f"missing update history: {update_history}")
    if not tensor_history.exists():
        raise RuntimeError(f"missing tensor norm history: {tensor_history}")
    if not activation_summary.exists():
        raise RuntimeError(f"missing activation probe summary: {activation_summary}")
    if not checkpoints:
        raise RuntimeError(f"missing checkpoints under {log_dir}")

    update_rows = [json.loads(line) for line in update_history.read_text(encoding="utf-8").splitlines() if line.strip()]
    optimizer_names = sorted({row.get("optimizer_name") for row in update_rows})
    if "ademamix" not in optimizer_names:
        raise RuntimeError(f"update history did not record AdEMAMix rows: {optimizer_names}")

    checkpoint = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    optimizer_states = checkpoint["optimizers"]
    if len(optimizer_states) != 2:
        raise RuntimeError(f"expected two optimizer states, found {len(optimizer_states)}")
    adema_state_values = list(optimizer_states[1]["state"].values())
    if not adema_state_values:
        raise RuntimeError("AdEMAMix optimizer state is empty")
    first_state = adema_state_values[0]
    expected_keys = {"step", "exp_avg_fast", "exp_avg_slow", "exp_avg_sq"}
    missing = expected_keys.difference(first_state)
    if missing:
        raise RuntimeError(f"AdEMAMix state missing keys: {sorted(missing)}")

    return {
        "log_dir": str(log_dir),
        "checkpoint_count": len(checkpoints),
        "last_checkpoint": str(checkpoints[-1]),
        "update_rows": len(update_rows),
        "optimizer_names": optimizer_names,
        "adema_state_keys": sorted(first_state.keys()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny AdEMAMix train_gpt2 smoke test.")
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=2)
    parser.add_argument("--val-tokens", type=int, default=64)
    parser.add_argument("--val-loss-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--warmdown-iters", type=int, default=1)
    parser.add_argument("--embed-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--adema-beta1", type=float, default=0.9)
    parser.add_argument("--adema-beta2", type=float, default=0.95)
    parser.add_argument("--adema-beta3", type=float, default=0.9999)
    parser.add_argument("--adema-alpha", type=float, default=8.0)
    parser.add_argument("--adema-beta3-warmup", type=int, default=0)
    parser.add_argument("--adema-alpha-warmup", type=int, default=0)
    parser.add_argument("--adema-eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("train_gpt2.py requires CUDA for this smoke")
    if args.val_tokens % (args.device_batch_size * args.sequence_length) != 0:
        raise ValueError("val_tokens must be divisible by device_batch_size * sequence_length")
    if args.batch_size % args.device_batch_size != 0:
        raise ValueError("batch_size must be divisible by device_batch_size")

    train_path, val_path = prepare_local_data(args.num_tokens, vocab_size=50304)
    train_pattern = str(train_path.parent / "fineweb_train_*.bin")
    val_pattern = str(val_path.parent / "fineweb_val_*.bin")
    render_temp_script(args, train_pattern, val_pattern)

    logs_dir = REPO_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    before = {path for path in logs_dir.glob("*") if path.is_dir()}
    command = [sys.executable, str(TEMP_SCRIPT)]
    completed = subprocess.run(command, cwd=REPO_DIR, check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"training smoke failed with exit code {completed.returncode}")

    log_dir = newest_log_dir(before)
    summary = inspect_outputs(log_dir)
    summary_path = LOCAL_DEBUG_DIR / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
