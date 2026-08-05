"""Plot exact LR schedules parsed from the cosine-wave training scripts."""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "wave_frequency_figures"
SCRIPTS = [
    (2, ROOT / "train_gpt2_gamma_cosine_wave_lrmatched_B0128_devB064.py"),
    (4, ROOT / "train_gpt2_gamma_cosine_wave4cycles_lrmatched_B0128_devB064.py"),
    (8, ROOT / "train_gpt2_gamma_cosine_wave8cycles_lrmatched_B0128_devB064.py"),
    (16, ROOT / "train_gpt2_gamma_cosine_wave16cycles_lrmatched_B0128_devB064.py"),
    (32, ROOT / "train_gpt2_gamma_cosine_wave32cycles_lrmatched_B0128_devB064.py"),
]
COLORS = {
    2: "#555555",
    4: "#1f77b4",
    8: "#2ca02c",
    16: "#ff7f0e",
    32: "#d62728",
}


def literal_assignments(nodes) -> dict[str, object]:
    values = {}
    for node in nodes:
        try:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                values[node.target.id] = ast.literal_eval(node.value)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                values[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return values


def parse_script(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hyperparameters = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Hyperparameters"
    )
    hparams = literal_assignments(hyperparameters.body)
    schedule = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "schedule_ratio"
    )
    schedule_values = literal_assignments(schedule.body)
    config_path = ROOT.parents[1] / hparams["norm_control_config"]
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    return {
        "num_iterations": int(hparams["num_iterations"]),
        "embed_learning_rate": float(hparams["embed_learning_rate"]),
        "warmup_iters": int(hparams["warmup_iters"]),
        "warmdown_iters": int(hparams["warmdown_iters"]),
        "start_step": int(config["start_step"]),
        "period_steps": int(schedule_values["period_steps"]),
    }


def build_rows(nominal_waves: int, path: Path) -> list[dict[str, float | int | str]]:
    cfg = parse_script(path)
    rows = []
    base_block_lr = 0.5 * cfg["embed_learning_rate"]
    actual_cycles = (cfg["num_iterations"] - cfg["start_step"]) / cfg["period_steps"]
    for step in range(cfg["num_iterations"] + 1):
        if step <= cfg["start_step"]:
            ratio = 1.0
        else:
            phase = 2.0 * math.pi * (step - cfg["start_step"]) / cfg["period_steps"]
            ratio = 1.0 + 0.5 * math.cos(phase - 0.5 * math.pi)
        if step < cfg["warmup_iters"]:
            wsd_ratio = (step + 1) / cfg["warmup_iters"]
        elif step < cfg["num_iterations"] - cfg["warmdown_iters"]:
            wsd_ratio = 1.0
        else:
            wsd_ratio = (cfg["num_iterations"] - step) / cfg["warmdown_iters"]
        rows.append({
            "script": path.name,
            "nominal_waves": nominal_waves,
            "period_steps": cfg["period_steps"],
            "actual_cycles_after_start": actual_cycles,
            "step": step,
            "schedule_ratio": ratio,
            "wsd_ratio": wsd_ratio,
            "block_lr": base_block_lr * wsd_ratio * ratio,
            "gamma_lr": base_block_lr * wsd_ratio,
        })
    return rows


def save_csv(rows: list[dict[str, object]]) -> Path:
    path = OUTPUT / "cosine_wave_lr_schedules.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_overlay(grouped: dict[int, list[dict[str, object]]]) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for waves, rows in grouped.items():
        steps = np.array([row["step"] for row in rows])
        lr = np.array([row["block_lr"] for row in rows])
        cycles = rows[0]["actual_cycles_after_start"]
        ax.plot(steps, lr, color=COLORS[waves], label=f"{waves} waves ({cycles:.2f} after step 1000)")
    gamma_lr = np.array([row["gamma_lr"] for row in grouped[2]])
    steps = np.array([row["step"] for row in grouped[2]])
    ax.plot(steps, gamma_lr, color="black", linestyle="--", linewidth=1.2, label="gamma LR (WSD only)")
    ax.axvline(1000, color="0.55", linestyle=":", linewidth=1.1)
    ax.axvline(14600, color="0.72", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 20400)
    ax.set_ylim(0, 0.00285)
    ax.set_xticks([0, 1000, 5000, 10000, 14600, 20000])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("block learning rate")
    ax.set_title("Actual block LR: WSD envelope × cosine norm-schedule ratio")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    path = OUTPUT / "block_lr_wave_frequency_overlay.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_small_multiples(grouped: dict[int, list[dict[str, object]]]) -> Path:
    fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex="col", constrained_layout=True)
    for row_index, (waves, rows) in enumerate(grouped.items()):
        steps = np.array([row["step"] for row in rows])
        lr = np.array([row["block_lr"] for row in rows])
        gamma_lr = np.array([row["gamma_lr"] for row in rows])
        ratio = np.array([row["schedule_ratio"] for row in rows])
        cycles = rows[0]["actual_cycles_after_start"]
        period = rows[0]["period_steps"]

        left, right = axes[row_index]
        left.plot(steps, lr, color=COLORS[waves])
        left.plot(steps, gamma_lr, color="black", linestyle="--", linewidth=1.0)
        left.axvline(1000, color="0.6", linestyle=":", linewidth=0.9)
        left.axvline(14600, color="0.75", linestyle=":", linewidth=0.9)
        left.set_ylim(0, 0.00285)
        left.set_ylabel("block LR")
        left.set_title(f"Nominal {waves} waves: period={period:,} steps")

        mask = steps >= 1000
        right.plot(steps[mask], ratio[mask], color=COLORS[waves])
        right.axhline(1.0, color="0.55", linestyle="--", linewidth=0.9)
        right.set_ylim(0.45, 1.55)
        right.set_ylabel("LR / WSD LR")
        right.set_title(f"Wave multiplier ({cycles:.2f} cycles after control start)")

    for ax in axes[-1]:
        ax.set_xlabel("optimizer step")
    axes[0, 0].legend(["block LR", "gamma LR (WSD only)"], frameon=False, loc="upper right")
    path = OUTPUT / "block_lr_wave_frequency_small_multiples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    grouped = {waves: build_rows(waves, path) for waves, path in SCRIPTS}
    rows = [row for group in grouped.values() for row in group]
    csv_path = save_csv(rows)
    overlay_path = plot_overlay(grouped)
    multiples_path = plot_small_multiples(grouped)
    for path in (csv_path, overlay_path, multiples_path):
        print(path)


if __name__ == "__main__":
    main()
