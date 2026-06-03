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
        experiment_id="try_m099_wd0p7",
        role="trial",
        muon_momentum=0.99,
        muon_nesterov=False,
        muon_weight_decay=0.7,
        script_name="train_try_m099_wd0p7.py",
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


CONTROL_EXPERIMENTS = [
    dict(
        experiment_id="muwdsched_exp_start6_floor1p1_tau2000",
        role="trial + bang-bang Muon weight decay control",
        muon_momentum=0.99,
        muon_nesterov=False,
        muon_weight_decay_control=dict(initial_weight_decay=1.0, multiplier=1.05, lower_ratio=0.97, upper_ratio=1.03),
        script_name="train_muwdsched_exp_start6_floor1p1_tau2000.py",
    ),
    dict(
        experiment_id="muwdsched_exp_start6_floor1p1_tau1200",
        role="trial + bang-bang Muon weight decay control",
        muon_momentum=0.99,
        muon_nesterov=False,
        muon_weight_decay_control=dict(initial_weight_decay=1.0, multiplier=1.05, lower_ratio=0.97, upper_ratio=1.03),
        script_name="train_muwdsched_exp_start6_floor1p1_tau1200.py",
    ),
]


LEGACY_WEIGHT_DECAY_LINE = "    muon_weight_decay : float = 0 # decoupled weight decay for Muon parameters"

CONTROL_TEMPLATE = (
    "MUON_WEIGHT_DECAY_CONTROL = dict(\n"
    "    initial_weight_decay=1.0,\n"
    "    multiplier=1.05,\n"
    "    lower_ratio=0.97,\n"
    "    upper_ratio=1.03,\n"
    ")"
)


def replace_exact(source, old, new):
    if old not in source:
        raise RuntimeError(f"missing template line: {old}")
    return source.replace(old, new, 1)


def float_literal(value):
    return repr(float(value))


def format_control(control):
    return (
        "MUON_WEIGHT_DECAY_CONTROL = dict(\n"
        f"    initial_weight_decay={float_literal(control['initial_weight_decay'])},\n"
        f"    multiplier={float_literal(control['multiplier'])},\n"
        f"    lower_ratio={float_literal(control['lower_ratio'])},\n"
        f"    upper_ratio={float_literal(control['upper_ratio'])},\n"
        ")"
    )


def apply_momentum_settings(code, experiment):
    momentum = float_literal(experiment["muon_momentum"])
    nesterov = str(bool(experiment["muon_nesterov"]))
    code = replace_exact(
        code,
        "    muon_momentum : float = 0.95 # Muon momentum beta",
        f"    muon_momentum : float = {momentum} # Muon momentum beta",
    )
    return replace_exact(
        code,
        "    muon_nesterov : bool = False # whether to use Nesterov-style Muon momentum",
        f"    muon_nesterov : bool = {nesterov} # whether to use Nesterov-style Muon momentum",
    )


def main():
    experiment_dir = Path(__file__).resolve().parent
    repo_root = experiment_dir.parents[1]
    template_path = repo_root / "train_gpt2.py"
    output_dir = experiment_dir / "generated_train_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    template = template_path.read_text(encoding="utf-8")
    for experiment in EXPERIMENTS:
        if LEGACY_WEIGHT_DECAY_LINE not in template:
            continue
        code = template
        weight_decay = float_literal(experiment["muon_weight_decay"])
        code = apply_momentum_settings(code, experiment)
        code = replace_exact(
            code,
            LEGACY_WEIGHT_DECAY_LINE,
            f"    muon_weight_decay : float = {weight_decay} # decoupled weight decay for Muon parameters",
        )

        script_path = output_dir / experiment["script_name"]
        script_path.write_text(code, encoding="utf-8")

    for experiment in CONTROL_EXPERIMENTS:
        code = apply_momentum_settings(template, experiment)
        code = replace_exact(
            code,
            CONTROL_TEMPLATE,
            format_control(experiment["muon_weight_decay_control"]),
        )

        script_path = output_dir / experiment["script_name"]
        script_path.write_text(code, encoding="utf-8")


if __name__ == "__main__":
    main()
