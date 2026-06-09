from __future__ import annotations

import ast
import fnmatch
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO_DIR = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_DIR / "train_gpt2.py"
CONFIG = REPO_DIR / "experiments" / "norm_control_optimizer" / "tied_embedding_norm_control_smoke.json"

FUNCTIONS = {
    "canonical_param_name",
    "pattern_matches_name",
    "is_allowed_norm_control_parameter",
    "load_norm_control_config",
    "build_norm_control_state",
    "apply_rms_norm_control",
    "write_norm_control_targets",
    "write_norm_control_metadata",
}


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.c_q = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(4, 8, bias=False)


class TinyGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(16, 4)
        self.transformer.h = nn.ModuleList([TinyBlock()])
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.transformer.wte.weight = self.lm_head.weight


class CompiledNameWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module


def load_train_norm_control_functions() -> dict[str, object]:
    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    missing = FUNCTIONS - {node.name for node in selected}
    if missing:
        raise RuntimeError(f"missing expected functions in {TRAIN_SCRIPT}: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "fnmatch": fnmatch,
        "json": json,
        "os": os,
        "torch": torch,
        "args": SimpleNamespace(num_iterations=1),
        "master_process": True,
        "DEFAULT_NORM_CONTROL_MODE": "delayed_captured_constant",
        "DEFAULT_NORM_CONTROL_START_STEP": 1000,
    }
    exec(compile(module, str(TRAIN_SCRIPT), "exec"), namespace)
    return namespace


def main() -> None:
    namespace = load_train_norm_control_functions()
    model = CompiledNameWrapper(TinyGPT())
    with tempfile.TemporaryDirectory(prefix="tied_embedding_norm_control_") as tmpdir:
        namespace["logdir"] = tmpdir
        state = namespace["build_norm_control_state"](model, str(CONFIG))
        controlled_names = [entry["name"] for entry in state["params"]]
        tied_names = [name for name in controlled_names if name.endswith("transformer.wte.weight")]
        if tied_names != ["_orig_mod.transformer.wte.weight"]:
            raise AssertionError(f"expected one tied embedding target, got {controlled_names}")
        namespace["apply_rms_norm_control"](state, step=0, event="initial")
        namespace["write_norm_control_metadata"](state)

        metadata = json.loads((Path(tmpdir) / "norm_control_metadata.json").read_text())
        history_rows = [
            json.loads(line)
            for line in (Path(tmpdir) / "norm_control_history.jsonl").read_text().splitlines()
        ]
        metadata_names = {record["name"] for record in metadata["controlled_parameters"]}
        history_names = {record["name"] for record in history_rows}
        tied_name = "_orig_mod.transformer.wte.weight"
        if tied_name not in metadata_names or tied_name not in history_names:
            raise AssertionError("tied embedding missing from metadata/history")
        tied_rows = [record for record in history_rows if record["name"] == tied_name]
        if tied_rows[0]["event"] != "initial" or tied_rows[0]["projected"] is not True:
            raise AssertionError(f"tied embedding initial projection row is wrong: {tied_rows[0]}")
        if tied_rows[0]["relative_error"] > 1e-5:
            raise AssertionError(f"tied embedding RMS projection error too large: {tied_rows[0]}")
        if any(record["weight_decay"] != 0.0 for record in history_rows):
            raise AssertionError("controlled tensors should log weight_decay=0.0")
        print(json.dumps({
            "status": "ok",
            "controlled_names": controlled_names,
            "history_rows": len(history_rows),
            "tied_embedding_relative_error": tied_rows[0]["relative_error"],
        }, indent=2))


if __name__ == "__main__":
    main()
