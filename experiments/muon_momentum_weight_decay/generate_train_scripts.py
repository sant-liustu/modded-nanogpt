from pathlib import Path


EXPERIMENTS = [
    dict(
        experiment_id="muwdcontrol_target_m095_muwd0p1_mult1p05_deadband3p",
        role="trial + local target-norm Muon weight decay controller",
        muon_momentum=0.99,
        muon_nesterov=False,
        script_name="train_muwdcontrol_target_m095_muwd0p1_mult1p05_deadband3p.py",
    ),
]


def replace_exact(source, old, new):
    if old not in source:
        raise RuntimeError(f"missing template line: {old}")
    return source.replace(old, new, 1)


def main():
    experiment_dir = Path(__file__).resolve().parent
    repo_root = experiment_dir.parents[1]
    template_path = repo_root / "train_gpt2.py"
    output_dir = experiment_dir / "generated_train_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    template = template_path.read_text(encoding="utf-8")
    for experiment in EXPERIMENTS:
        code = template
        momentum = repr(float(experiment["muon_momentum"]))
        nesterov = str(bool(experiment["muon_nesterov"]))
        code = replace_exact(
            code,
            "    muon_momentum : float = 0.95 # Muon momentum beta",
            f"    muon_momentum : float = {momentum} # Muon momentum beta",
        )
        code = replace_exact(
            code,
            "    muon_nesterov : bool = False # whether to use Nesterov-style Muon momentum",
            f"    muon_nesterov : bool = {nesterov} # whether to use Nesterov-style Muon momentum",
        )

        script_path = output_dir / experiment["script_name"]
        script_path.write_text(code, encoding="utf-8")


if __name__ == "__main__":
    main()
