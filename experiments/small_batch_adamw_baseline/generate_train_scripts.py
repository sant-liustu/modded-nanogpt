from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_DIR = EXPERIMENT_DIR.parents[1]
TEMPLATE = REPO_DIR / "train_gpt2.py"
OUTPUT_DIR = EXPERIMENT_DIR / "generated_train_scripts"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    device_batch_size: int = 64
    sequence_length: int = 1024
    num_iterations: int = 20400
    embed_learning_rate: float = 0.0036
    warmup_iters: int = 1000
    warmdown_iters: int = 5800
    weight_decay: float = 0.0
    val_loss_every: int = 500
    val_tokens: int = 10485760
    save_every: int = 0
    compile_model: int = 1
    tensor_norm_every: int = 4
    adamw_update_norm_every: int = 4
    activation_probe_every: int = 0
    spectral_norm_estimate_enabled: int = 0
    seed: int = 0

    @property
    def filename(self) -> str:
        return (
            "train_small_batch_adamw_"
            f"B{self.batch_size:04d}_"
            f"devB{self.device_batch_size:03d}_"
            f"lr{format_tag(self.embed_learning_rate)}_"
            f"wd{format_tag(self.weight_decay)}_"
            f"seed{self.seed:02d}.py"
        )


FIELD_TYPES = {
    "batch_size": "int",
    "device_batch_size": "int",
    "sequence_length": "int",
    "num_iterations": "int",
    "embed_learning_rate": "float",
    "muon_learning_rate": "float",
    "warmup_iters": "int",
    "warmdown_iters": "int",
    "weight_decay": "float",
    "val_loss_every": "int",
    "val_tokens": "int",
    "save_every": "int",
    "compile_model": "int",
    "tensor_norm_every": "int",
    "adamw_update_norm_every": "int",
    "activation_probe_every": "int",
    "spectral_norm_estimate_enabled": "int",
    "activation_probe_eps": "float",
    "norm_control_config": "str",
    "seed": "int",
}


def format_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def format_value(value: object) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


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


def insert_seed_field(lines: list[str], seed: int) -> None:
    if any(re.match(r"^\s*seed\s*:", line) for line in lines):
        replace_field(lines, "seed", seed)
        return

    marker = "    norm_control_config : str ="
    matches = [idx for idx, line in enumerate(lines) if line.startswith(marker)]
    if len(matches) != 1:
        raise RuntimeError("could not find norm_control_config line for seed insertion")
    lines.insert(matches[0] + 1, f"    seed : int = {seed} # RNG seed for generated experiment scripts\n")


def insert_seed_setup(lines: list[str]) -> None:
    marker = "args = Hyperparameters()\n"
    matches = [idx for idx, line in enumerate(lines) if line == marker]
    if len(matches) != 1:
        raise RuntimeError("could not find args = Hyperparameters() for seed setup")
    idx = matches[0] + 1
    seed_block = [
        "\n",
        "# experiment-generated seed control\n",
        "torch.manual_seed(args.seed)\n",
        "np.random.seed(args.seed)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(args.seed)\n",
    ]
    lines[idx:idx] = seed_block


def render_script(template: str, config: TrainConfig) -> str:
    lines = template.splitlines(keepends=True)
    replacements = {
        "batch_size": config.batch_size,
        "device_batch_size": config.device_batch_size,
        "sequence_length": config.sequence_length,
        "num_iterations": config.num_iterations,
        "embed_learning_rate": config.embed_learning_rate,
        "warmup_iters": config.warmup_iters,
        "warmdown_iters": config.warmdown_iters,
        "weight_decay": config.weight_decay,
        "val_loss_every": config.val_loss_every,
        "val_tokens": config.val_tokens,
        "save_every": config.save_every,
        "compile_model": config.compile_model,
        "tensor_norm_every": config.tensor_norm_every,
        "adamw_update_norm_every": config.adamw_update_norm_every,
        "activation_probe_every": config.activation_probe_every,
        "spectral_norm_estimate_enabled": config.spectral_norm_estimate_enabled,
        "norm_control_config": "",
    }
    for name, value in replacements.items():
        replace_field(lines, name, value)
    insert_seed_field(lines, config.seed)
    insert_seed_setup(lines)
    header = [
        "# Generated by experiments/small_batch_adamw_baseline/generate_train_scripts.py.\n",
        "# Do not hand edit; regenerate this file from the template training script.\n",
        "# Purpose: standard AdamW baseline at B=128, T=1024, 20400 steps.\n",
        "# Token budget matches the B=512, T=1024, 5100-step large-batch setup.\n",
        f"# Config: batch_size={config.batch_size}, device_batch_size={config.device_batch_size}, "
        f"sequence_length={config.sequence_length}, num_iterations={config.num_iterations}, "
        f"lr={config.embed_learning_rate}, block_weight_decay={config.weight_decay}, seed={config.seed}\n",
        "\n",
    ]
    return "".join(header + lines)


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"missing template training script: {TEMPLATE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_script in OUTPUT_DIR.glob("train_*.py"):
        old_script.unlink()

    config = TrainConfig()
    output_path = OUTPUT_DIR / config.filename
    template = TEMPLATE.read_text(encoding="utf-8")
    output_path.write_text(render_script(template, config), encoding="utf-8", newline="\n")
    print(output_path)


if __name__ == "__main__":
    main()
