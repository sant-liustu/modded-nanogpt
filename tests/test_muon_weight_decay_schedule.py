import ast
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = REPO_ROOT / "train_gpt2.py"


def load_schedule_namespace():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    wanted = {
        "MUON_WEIGHT_DECAY_SCHEDULE",
        "muon_weight_decay_for_step",
        "set_muon_weight_decay_for_step",
    }
    nodes = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_SOURCE), "exec"), namespace)
    return namespace


def test_muon_weight_decay_schedule_values():
    ns = load_schedule_namespace()
    schedule = ns["MUON_WEIGHT_DECAY_SCHEDULE"]
    wd_for_step = ns["muon_weight_decay_for_step"]

    assert wd_for_step(0) == schedule["start"]
    assert wd_for_step(schedule["hold_steps"] - 1) == schedule["start"]
    assert wd_for_step(schedule["hold_steps"]) == schedule["start"]
    assert wd_for_step(schedule["decay_end"]) == schedule["floor"]
    assert wd_for_step(5100) == schedule["floor"]

    midpoint = (schedule["hold_steps"] + schedule["decay_end"]) // 2
    expected_midpoint = schedule["floor"] + 0.5 * (schedule["start"] - schedule["floor"])
    assert math.isclose(wd_for_step(midpoint), expected_midpoint)
    assert wd_for_step(1000) > wd_for_step(1500)


def test_set_muon_weight_decay_updates_all_param_groups():
    ns = load_schedule_namespace()

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"weight_decay": -1.0}, {"weight_decay": -2.0}]

    ns["optimizer2"] = FakeOptimizer()
    weight_decay = ns["set_muon_weight_decay_for_step"](1000)

    assert math.isclose(ns["optimizer2"].param_groups[0]["weight_decay"], weight_decay)
    assert math.isclose(ns["optimizer2"].param_groups[1]["weight_decay"], weight_decay)


def test_muon_step_reads_current_group_weight_decay():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    read_pos = source.index("            weight_decay = group['weight_decay']")
    apply_pos = source.index("                    p.data.mul_(1 - lr * weight_decay)")

    assert read_pos < apply_pos


def test_schedule_is_set_before_update_state_capture():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    update_step_pos = source.index("    update_step = step + 1")
    set_pos = source.index("    set_muon_weight_decay_for_step(step)")
    capture_pos = source.index("    optimizer_update_state = maybe_capture_optimizer_update_state(update_step)")

    assert update_step_pos < set_pos < capture_pos


if __name__ == "__main__":
    test_muon_weight_decay_schedule_values()
    test_set_muon_weight_decay_updates_all_param_groups()
    test_muon_step_reads_current_group_weight_decay()
    test_schedule_is_set_before_update_state_capture()
