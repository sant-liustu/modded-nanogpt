import ast
import json
import math
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = REPO_ROOT / "train_gpt2.py"
TARGET_NORM_FILE = (
    REPO_ROOT
    / "experiments"
    / "muon_momentum_weight_decay"
    / "target_norm_mom0p95_muwd0p1_nestF_5000.json"
)


def load_controller_namespace():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    wanted = {"compare_muon_norm", "update_muon_weight_decay"}
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = dict(
        MUON_WD_MULTIPLIER=1.05,
        MUON_WD_DEADBAND=0.03,
        MUON_WD_MIN=0.0,
        MUON_WD_MAX=20.0,
        MUON_TARGET_NORMS=[1.0],
        muon_weight_decay=5.0,
        torch=torch,
        json=json,
        os=__import__("os"),
        master_process=False,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_SOURCE), "exec"), namespace)
    return namespace


def test_local_target_norm_file_is_single_line_json():
    content = TARGET_NORM_FILE.read_text(encoding="utf-8")
    assert "\n" not in content
    values = json.loads(content)
    assert len(values) == 5000
    assert all(isinstance(value, float) and value > 0 for value in values)


def test_compare_muon_norm_uses_three_percent_deadband():
    ns = load_controller_namespace()
    compare = ns["compare_muon_norm"]

    assert compare(1.031, 1.0) == 1
    assert compare(0.969, 1.0) == -1
    assert compare(1.0, 1.0) == 0


def test_update_muon_weight_decay_updates_optimizer_groups():
    ns = load_controller_namespace()

    class FakeBlockStack:
        def __init__(self, value):
            self.value = value

        def parameters(self):
            return [torch.full((2, 2), self.value), torch.ones(2)]

    class FakeRawModel:
        def __init__(self, value):
            self.transformer = type("Transformer", (), {})()
            self.transformer.h = FakeBlockStack(value)

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"weight_decay": -1.0}, {"weight_decay": -2.0}]

    ns["raw_model"] = FakeRawModel(2.0)
    ns["optimizer2"] = FakeOptimizer()
    weight_decay = ns["update_muon_weight_decay"](0, 1)
    assert math.isclose(weight_decay, 5.25)
    assert all(math.isclose(group["weight_decay"], 5.25) for group in ns["optimizer2"].param_groups)

    ns["raw_model"] = FakeRawModel(0.5)
    ns["muon_weight_decay"] = 5.0
    ns["optimizer2"] = FakeOptimizer()
    weight_decay = ns["update_muon_weight_decay"](0, 1)
    assert math.isclose(weight_decay, 5.0 / 1.05)
    assert all(math.isclose(group["weight_decay"], weight_decay) for group in ns["optimizer2"].param_groups)


def test_muon_step_reads_current_group_weight_decay():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    read_pos = source.index("            weight_decay = group['weight_decay']")
    apply_pos = source.index("                    p.data.mul_(1 - lr * weight_decay)")

    assert read_pos < apply_pos


def test_weight_decay_is_updated_before_update_state_capture():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    update_step_pos = source.index("    update_step = step + 1")
    update_pos = source.index("    update_muon_weight_decay(step, update_step)")
    capture_pos = source.index("    optimizer_update_state = maybe_capture_optimizer_update_state(update_step)")

    assert update_step_pos < update_pos < capture_pos


if __name__ == "__main__":
    test_local_target_norm_file_is_single_line_json()
    test_compare_muon_norm_uses_three_percent_deadband()
    test_update_muon_weight_decay_updates_optimizer_groups()
    test_muon_step_reads_current_group_weight_decay()
    test_weight_decay_is_updated_before_update_state_capture()
