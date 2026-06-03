import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = REPO_ROOT / "train_gpt2.py"


def load_schedule_namespace():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    wanted = {
        "MUON_WEIGHT_DECAY_CONTROL",
        "MUON_WEIGHT_DECAY_CONTROL_STATE",
        "set_muon_weight_decay",
        "muon_parameter_rms_norm",
        "update_muon_weight_decay_for_current_norm",
    }
    nodes = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_SOURCE), "exec"), namespace)
    return namespace


def test_muon_weight_decay_control_values():
    ns = load_schedule_namespace()
    control = ns["MUON_WEIGHT_DECAY_CONTROL"]
    state = ns["MUON_WEIGHT_DECAY_CONTROL_STATE"]

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"weight_decay": -1.0}, {"weight_decay": -2.0}]

    ns["optimizer2"] = FakeOptimizer()
    state["target_norm"] = 10.0

    state["current_weight_decay"] = 1.0
    ns["muon_parameter_rms_norm"] = lambda: 10.4
    assert ns["update_muon_weight_decay_for_current_norm"]() == 1.0 * control["multiplier"]

    state["current_weight_decay"] = 1.0
    ns["muon_parameter_rms_norm"] = lambda: 9.6
    assert ns["update_muon_weight_decay_for_current_norm"]() == 1.0 / control["multiplier"]

    state["current_weight_decay"] = 1.0
    ns["muon_parameter_rms_norm"] = lambda: 10.0
    assert ns["update_muon_weight_decay_for_current_norm"]() == 1.0


def test_set_muon_weight_decay_updates_all_param_groups():
    ns = load_schedule_namespace()

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"weight_decay": -1.0}, {"weight_decay": -2.0}]

    ns["optimizer2"] = FakeOptimizer()
    weight_decay = ns["set_muon_weight_decay"](1.25)

    assert ns["optimizer2"].param_groups[0]["weight_decay"] == weight_decay
    assert ns["optimizer2"].param_groups[1]["weight_decay"] == weight_decay


def test_muon_step_reads_current_group_weight_decay():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    read_pos = source.index("            weight_decay = group['weight_decay']")
    apply_pos = source.index("                    p.data.mul_(1 - lr * weight_decay)")

    assert read_pos < apply_pos


def test_schedule_is_set_before_update_state_capture():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    update_step_pos = source.index("    update_step = step + 1")
    set_pos = source.index("    update_muon_weight_decay_for_current_norm()")
    capture_pos = source.index("    optimizer_update_state = maybe_capture_optimizer_update_state(update_step)")

    assert update_step_pos < set_pos < capture_pos


if __name__ == "__main__":
    test_muon_weight_decay_control_values()
    test_set_muon_weight_decay_updates_all_param_groups()
    test_muon_step_reads_current_group_weight_decay()
    test_schedule_is_set_before_update_state_capture()
