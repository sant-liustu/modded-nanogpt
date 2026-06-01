from pathlib import Path


EXPERIMENTS = [
    dict(
        experiment_id="baseline_m095_wd0",
        role="baseline",
        muon_momentum=0.95,
        muon_nesterov=False,
        muon_weight_decay=0.0,
        script_name="train_baseline_m095_wd0.py",
    ),
    dict(
        experiment_id="baseline_m095_wd0p1",
        role="baseline + weight decay",
        muon_momentum=0.95,
        muon_nesterov=False,
        muon_weight_decay=0.1,
        script_name="train_baseline_m095_wd0p1.py",
    ),
    dict(
        experiment_id="try_m099_wd0p1",
        role="trial",
        muon_momentum=0.99,
        muon_nesterov=False,
        muon_weight_decay=0.1,
        script_name="train_try_m099_wd0p1.py",
    ),
    dict(
        experiment_id="try_m099_wd0p2",
        role="trial",
        muon_momentum=0.99,
        muon_nesterov=False,
        muon_weight_decay=0.2,
        script_name="train_try_m099_wd0p2.py",
    ),
    dict(
        experiment_id="try_m099_wd0",
        role="momentum-only control",
        muon_momentum=0.99,
        muon_nesterov=False,
        muon_weight_decay=0.0,
        script_name="train_try_m099_wd0.py",
    ),
    dict(
        experiment_id="baseline_m095_wd0_nesterov",
        role="baseline + Nesterov",
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_weight_decay=0.0,
        script_name="train_baseline_m095_wd0_nesterov.py",
    ),
    dict(
        experiment_id="baseline_m095_wd0p1_nesterov",
        role="baseline + weight decay + Nesterov",
        muon_momentum=0.95,
        muon_nesterov=True,
        muon_weight_decay=0.1,
        script_name="train_baseline_m095_wd0p1_nesterov.py",
    ),
    dict(
        experiment_id="try_m099_wd0p1_nesterov",
        role="trial + Nesterov",
        muon_momentum=0.99,
        muon_nesterov=True,
        muon_weight_decay=0.1,
        script_name="train_try_m099_wd0p1_nesterov.py",
    ),
    dict(
        experiment_id="try_m099_wd0p2_nesterov",
        role="trial + Nesterov",
        muon_momentum=0.99,
        muon_nesterov=True,
        muon_weight_decay=0.2,
        script_name="train_try_m099_wd0p2_nesterov.py",
    ),
    dict(
        experiment_id="try_m099_wd0_nesterov",
        role="momentum-only control + Nesterov",
        muon_momentum=0.99,
        muon_nesterov=True,
        muon_weight_decay=0.0,
        script_name="train_try_m099_wd0_nesterov.py",
    ),
]


def replace_exact(source, old, new):
    if old not in source:
        raise RuntimeError(f"missing template line: {old}")
    return source.replace(old, new, 1)


def float_literal(value):
    return repr(float(value))


def main():
    experiment_dir = Path(__file__).resolve().parent
    repo_root = experiment_dir.parents[1]
    template_path = repo_root / "train_gpt2.py"
    output_dir = experiment_dir / "generated_train_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    template = template_path.read_text(encoding="utf-8")
    for experiment in EXPERIMENTS:
        code = template
        momentum = float_literal(experiment["muon_momentum"])
        nesterov = str(bool(experiment["muon_nesterov"]))
        weight_decay = float_literal(experiment["muon_weight_decay"])
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
        code = replace_exact(
            code,
            "    muon_weight_decay : float = 0 # decoupled weight decay for Muon parameters",
            f"    muon_weight_decay : float = {weight_decay} # decoupled weight decay for Muon parameters",
        )

        script_path = output_dir / experiment["script_name"]
        script_path.write_text(code, encoding="utf-8")


if __name__ == "__main__":
    main()
