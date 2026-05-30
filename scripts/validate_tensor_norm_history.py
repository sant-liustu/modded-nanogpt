import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


ACTIVATION_PROBE_FIELDS = {
    "rms_h_pre",
    "attn_residual_ratio",
    "attn_branch_ratio",
    "mlp_residual_ratio",
    "mlp_branch_ratio",
}


def load_jsonl(path):
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def is_finite_number(value):
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))


def required_norm_fields(ndim, prefix=""):
    if ndim == 2:
        return {f"{prefix}fro_norm", f"{prefix}rms_norm"}
    if ndim == 1:
        return {f"{prefix}fro_norm", f"{prefix}rms_norm"}
    if ndim == 0:
        return {f"{prefix}abs_value", f"{prefix}rms_norm"}
    return {f"{prefix}fro_norm", f"{prefix}rms_norm"}


def validate_history_file(path, trainable, errors, prefix="", require_optimizer_fields=False):
    seen_by_tensor = defaultdict(list)
    rows = 0
    for line_number, record in load_jsonl(path):
        rows += 1
        name = record.get("name")
        if name not in trainable:
            errors.append(f"{path.name} line {line_number}: unknown tensor {name}")
            continue
        ndim = trainable[name]["ndim"]
        if record.get("ndim") != ndim:
            errors.append(f"{path.name} line {line_number}: ndim mismatch for {name}")
        if record.get("shape") != trainable[name]["shape"]:
            errors.append(f"{path.name} line {line_number}: shape mismatch for {name}")
        for key, value in record.items():
            is_norm_field = (
                key.endswith("_norm")
                or key.endswith("_norm_estimate")
                or key.endswith("_abs_value")
                or key == "abs_value"
            )
            if is_norm_field:
                if not is_finite_number(value):
                    errors.append(f"{path.name} line {line_number}: non-finite or non-numeric {key} for {name}")
        spectral_keys = {
            f"{prefix}spectral_norm_estimate",
            f"{prefix}spectral_norm_estimate_method",
            f"{prefix}spectral_norm_estimate_block_size",
            f"{prefix}spectral_norm_estimate_iters",
        }
        present_spectral_keys = spectral_keys & set(record)
        required_fields = required_norm_fields(ndim, prefix)
        if present_spectral_keys:
            required_fields |= spectral_keys
        missing = sorted(required_fields - set(record))
        if missing:
            errors.append(f"{path.name} line {line_number}: missing {missing} for {name}")
        if ndim == 2 and present_spectral_keys:
            method_key = f"{prefix}spectral_norm_estimate_method"
            block_size_key = f"{prefix}spectral_norm_estimate_block_size"
            iters_key = f"{prefix}spectral_norm_estimate_iters"
            if not isinstance(record.get(method_key), str) or not record.get(method_key):
                errors.append(f"{path.name} line {line_number}: missing or invalid {method_key} for {name}")
            if not isinstance(record.get(block_size_key), int) or record.get(block_size_key) <= 0:
                errors.append(f"{path.name} line {line_number}: missing or invalid {block_size_key} for {name}")
            if not isinstance(record.get(iters_key), int) or record.get(iters_key) <= 0:
                errors.append(f"{path.name} line {line_number}: missing or invalid {iters_key} for {name}")
        if require_optimizer_fields:
            for key in ("lr", "weight_decay"):
                if not is_finite_number(record.get(key)):
                    errors.append(f"{path.name} line {line_number}: missing or non-finite {key} for {name}")
            for key in ("optimizer_index", "param_group_index"):
                if not isinstance(record.get(key), int):
                    errors.append(f"{path.name} line {line_number}: missing or non-integer {key} for {name}")
        seen_by_tensor[name].append(record.get("step"))

    missing_tensors = sorted(set(trainable) - set(seen_by_tensor))
    if missing_tensors:
        errors.append(f"{path.name}: missing history for tensors: {missing_tensors}")

    counts = Counter(len(steps) for steps in seen_by_tensor.values())
    distinct_steps = sorted({step for steps in seen_by_tensor.values() for step in steps})
    return {
        "tensors": len(seen_by_tensor),
        "rows": rows,
        "distinct_steps": distinct_steps,
        "min_records_per_tensor": min(counts) if counts else 0,
        "max_records_per_tensor": max(counts) if counts else 0,
    }


def validate_activation_probe(run_dir, errors):
    metadata_path = run_dir / "activation_probe_metadata.json"
    summary_path = run_dir / "activation_probe_summary.jsonl"
    arrays_dir = run_dir / "activation_probe_arrays"

    if not metadata_path.exists():
        if summary_path.exists() or arrays_dir.exists():
            errors.append(f"missing activation probe metadata file: {metadata_path}")
        return {"activation_probe_steps": [], "activation_probe_summary_rows": 0, "activation_probe_array_files": 0}
    if not summary_path.exists():
        errors.append(f"missing activation probe summary file: {summary_path}")
    if not arrays_dir.exists():
        errors.append(f"missing activation probe arrays directory: {arrays_dir}")
        return {"activation_probe_steps": [], "activation_probe_summary_rows": 0, "activation_probe_array_files": 0}

    metadata = json.loads(metadata_path.read_text())
    recorded_fields = set(metadata.get("recorded_fields", []))
    missing_fields = sorted(ACTIVATION_PROBE_FIELDS - recorded_fields)
    extra_fields = sorted(recorded_fields - ACTIVATION_PROBE_FIELDS)
    if missing_fields:
        errors.append(f"activation metadata missing fields: {missing_fields}")
    if extra_fields:
        errors.append(f"activation metadata has unexpected fields: {extra_fields}")
    if metadata.get("array_layout") != "[layer, batch, seq]":
        errors.append("activation metadata array_layout must be [layer, batch, seq]")
    eps = metadata.get("eps")
    if not is_finite_number(eps) or eps < 0:
        errors.append("activation metadata eps must be a non-negative finite number")
        eps = 0.0
    layer_count = metadata.get("layer_count")
    if not isinstance(layer_count, int) or layer_count <= 0:
        errors.append("activation metadata layer_count must be a positive integer")
        layer_count = None
    probe_batch_shape = metadata.get("probe_batch_shape")
    if not (
        isinstance(probe_batch_shape, list)
        and len(probe_batch_shape) == 2
        and all(isinstance(dim, int) and dim > 0 for dim in probe_batch_shape)
    ):
        errors.append("activation metadata probe_batch_shape must be [batch, seq]")
        probe_batch_shape = None
    if not isinstance(metadata.get("probe_token_sha256"), str) or len(metadata.get("probe_token_sha256", "")) != 64:
        errors.append("activation metadata probe_token_sha256 must be a SHA256 hex string")

    summary_rows = 0
    summary_seen = defaultdict(set)
    if summary_path.exists():
        for line_number, record in load_jsonl(summary_path):
            summary_rows += 1
            step = record.get("step")
            layer = record.get("layer")
            field = record.get("field")
            if not isinstance(step, int) or step < 0:
                errors.append(f"{summary_path.name} line {line_number}: invalid step")
            if not isinstance(layer, int) or layer < 0 or (layer_count is not None and layer >= layer_count):
                errors.append(f"{summary_path.name} line {line_number}: invalid layer")
            if field not in ACTIVATION_PROBE_FIELDS:
                errors.append(f"{summary_path.name} line {line_number}: invalid field {field}")
            else:
                summary_seen[(step, field)].add(layer)
            for key in ("mean", "std", "p05", "p50", "p95", "min", "max"):
                if not is_finite_number(record.get(key)):
                    errors.append(f"{summary_path.name} line {line_number}: non-finite {key}")
            for key in ("nan_count", "inf_count"):
                if not isinstance(record.get(key), int) or record.get(key) < 0:
                    errors.append(f"{summary_path.name} line {line_number}: invalid {key}")

    steps = []
    array_files = sorted(arrays_dir.glob("step_*.pt"))
    for array_file in array_files:
        try:
            data = torch.load(array_file, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(array_file, map_location="cpu")
        step = data.get("step")
        if not isinstance(step, int) or step < 0:
            errors.append(f"{array_file.name}: missing or invalid step")
            continue
        steps.append(step)
        arrays = {}
        expected_shape = None
        for field in ACTIVATION_PROBE_FIELDS:
            value = data.get(field)
            if not isinstance(value, torch.Tensor):
                errors.append(f"{array_file.name}: missing tensor field {field}")
                continue
            if value.ndim != 3:
                errors.append(f"{array_file.name}: {field} must have shape [layer, batch, seq]")
                continue
            if layer_count is not None and value.shape[0] != layer_count:
                errors.append(f"{array_file.name}: {field} layer count mismatch")
            if probe_batch_shape is not None and list(value.shape[1:]) != probe_batch_shape:
                errors.append(f"{array_file.name}: {field} probe batch shape mismatch")
            if expected_shape is None:
                expected_shape = tuple(value.shape)
            elif tuple(value.shape) != expected_shape:
                errors.append(f"{array_file.name}: {field} shape does not match other activation fields")
            if not torch.isfinite(value).all().item():
                errors.append(f"{array_file.name}: {field} contains non-finite values")
            arrays[field] = value.float()

        if set(arrays) == ACTIVATION_PROBE_FIELDS:
            rms_h_pre = arrays["rms_h_pre"]
            rms_h_mid = arrays["attn_residual_ratio"] * (rms_h_pre + eps)
            rms_h_post = arrays["mlp_residual_ratio"] * (rms_h_mid + eps)
            if rms_h_pre.shape[0] > 1:
                expected_next = rms_h_pre[1:]
                reconstruction_error = (rms_h_post[:-1] - expected_next).abs()
                tolerance = 1e-3 + 1e-3 * expected_next.abs()
                if (reconstruction_error > tolerance).any().item():
                    errors.append(
                        f"{array_file.name}: reconstructed h_post[l] does not match rms_h_pre[l+1]; "
                        f"max_error={reconstruction_error.max().item():.6g}"
                    )
        if layer_count is not None:
            for field in ACTIVATION_PROBE_FIELDS:
                if summary_seen.get((step, field)) != set(range(layer_count)):
                    errors.append(f"{summary_path.name}: missing summary rows for step {step}, field {field}")

    if not array_files:
        errors.append("activation probe arrays directory has no step_*.pt files")

    return {
        "activation_probe_steps": sorted(steps),
        "activation_probe_summary_rows": summary_rows,
        "activation_probe_array_files": len(array_files),
    }


def close_enough(actual, expected, rtol=1e-6, atol=1e-12):
    if not is_finite_number(actual) or not is_finite_number(expected):
        return False
    return abs(actual - expected) <= atol + rtol * abs(expected)


def validate_norm_control(run_dir, trainable, errors, rms_rtol):
    metadata_path = run_dir / "norm_control_metadata.json"
    history_path = run_dir / "norm_control_history.jsonl"
    targets_path = run_dir / "norm_control_targets.json"
    if not metadata_path.exists() and not history_path.exists():
        return {
            "norm_control_enabled": False,
            "norm_control_mode": None,
            "norm_control_history_rows": 0,
            "norm_control_controlled_tensors": 0,
        }
    if not metadata_path.exists():
        errors.append(f"missing norm-control metadata file: {metadata_path}")
        return {
            "norm_control_enabled": False,
            "norm_control_mode": None,
            "norm_control_history_rows": 0,
            "norm_control_controlled_tensors": 0,
        }
    if not history_path.exists():
        errors.append(f"missing norm-control history file: {history_path}")
        return {
            "norm_control_enabled": True,
            "norm_control_mode": None,
            "norm_control_history_rows": 0,
            "norm_control_controlled_tensors": 0,
        }

    metadata = json.loads(metadata_path.read_text())
    mode = metadata.get("mode", "specified_target")
    controlled_records = metadata.get("controlled_parameters", [])
    controlled = {record.get("name"): record for record in controlled_records}
    if not controlled:
        errors.append("norm-control metadata has no controlled_parameters")
    for name, record in controlled.items():
        if name not in trainable:
            errors.append(f"norm-control metadata references unknown tensor {name}")
        if record.get("weight_decay") != 0.0:
            errors.append(f"norm-control controlled tensor {name} should have weight_decay=0.0")
        if mode == "specified_target":
            target_rms = record.get("target_rms")
            if not is_finite_number(target_rms) or target_rms <= 0:
                errors.append(f"norm-control metadata has invalid target_rms for {name}")

    start_step = metadata.get("start_step")
    if mode == "delayed_captured_constant":
        if not isinstance(start_step, int) or start_step < 0:
            errors.append("delayed norm-control metadata requires non-negative integer start_step")
            start_step = None
    elif mode != "specified_target":
        errors.append(f"unknown norm-control mode in metadata: {mode}")

    rows = 0
    seen_by_name = defaultdict(list)
    captured_by_name = {}
    projected_rows = 0
    max_relative_error = 0.0
    for line_number, record in load_jsonl(history_path):
        rows += 1
        name = record.get("name")
        if name not in controlled:
            errors.append(f"{history_path.name} line {line_number}: unknown controlled tensor {name}")
            continue
        seen_by_name[name].append(record)
        step = record.get("step")
        if not isinstance(step, int) or step < 0:
            errors.append(f"{history_path.name} line {line_number}: invalid step for {name}")
            continue
        if record.get("weight_decay") != 0.0:
            errors.append(f"{history_path.name} line {line_number}: controlled tensor {name} should log weight_decay=0.0")
        for key in ("pre_control_rms", "post_control_rms", "scale"):
            if not is_finite_number(record.get(key)):
                errors.append(f"{history_path.name} line {line_number}: missing or non-finite {key} for {name}")
        target_rms = record.get("target_rms")
        relative_error = record.get("relative_error")
        projected = record.get("projected")
        captured = record.get("captured")

        if mode == "specified_target":
            if record.get("mode") not in (None, "specified_target"):
                errors.append(f"{history_path.name} line {line_number}: wrong mode for specified-target run")
            if record.get("phase") not in (None, "specified_target"):
                errors.append(f"{history_path.name} line {line_number}: wrong phase for specified-target run")
            if projected is not True:
                errors.append(f"{history_path.name} line {line_number}: specified-target rows should be projected")
            if not is_finite_number(target_rms) or target_rms <= 0:
                errors.append(f"{history_path.name} line {line_number}: invalid target_rms for {name}")
            if not is_finite_number(relative_error) or relative_error > rms_rtol:
                errors.append(f"{history_path.name} line {line_number}: RMS relative_error too large for {name}: {relative_error}")
            elif is_finite_number(relative_error):
                max_relative_error = max(max_relative_error, relative_error)
            projected_rows += 1
            continue

        if start_step is None:
            continue
        phase = record.get("phase")
        if step < start_step:
            if phase != "pre_start":
                errors.append(f"{history_path.name} line {line_number}: expected pre_start phase before start_step")
            if projected is not False or captured is not False:
                errors.append(f"{history_path.name} line {line_number}: pre-start rows must not project or capture")
            if target_rms is not None or relative_error is not None:
                errors.append(f"{history_path.name} line {line_number}: pre-start rows should not have target_rms/relative_error")
            if is_finite_number(record.get("pre_control_rms")) and is_finite_number(record.get("post_control_rms")):
                if not close_enough(record["post_control_rms"], record["pre_control_rms"]):
                    errors.append(f"{history_path.name} line {line_number}: pre-start RMS changed for {name}")
        elif step == start_step:
            if phase != "capture":
                errors.append(f"{history_path.name} line {line_number}: expected capture phase at start_step")
            if projected is not False or captured is not True:
                errors.append(f"{history_path.name} line {line_number}: capture rows must capture without projection")
            if not is_finite_number(target_rms) or target_rms <= 0:
                errors.append(f"{history_path.name} line {line_number}: invalid captured target_rms for {name}")
            else:
                captured_by_name[name] = target_rms
                if not close_enough(record.get("pre_control_rms"), target_rms, rtol=rms_rtol):
                    errors.append(f"{history_path.name} line {line_number}: captured target does not match pre_control_rms for {name}")
            if is_finite_number(record.get("pre_control_rms")) and is_finite_number(record.get("post_control_rms")):
                if not close_enough(record["post_control_rms"], record["pre_control_rms"]):
                    errors.append(f"{history_path.name} line {line_number}: capture step RMS changed for {name}")
        else:
            if phase != "post_start":
                errors.append(f"{history_path.name} line {line_number}: expected post_start phase after start_step")
            if projected is not True or captured is not False:
                errors.append(f"{history_path.name} line {line_number}: post-start rows must project without capture")
            if not is_finite_number(target_rms) or target_rms <= 0:
                errors.append(f"{history_path.name} line {line_number}: invalid post-start target_rms for {name}")
            if not is_finite_number(relative_error) or relative_error > rms_rtol:
                errors.append(f"{history_path.name} line {line_number}: RMS relative_error too large for {name}: {relative_error}")
            elif is_finite_number(relative_error):
                max_relative_error = max(max_relative_error, relative_error)
            projected_rows += 1

    missing_history = sorted(set(controlled) - set(seen_by_name))
    if missing_history:
        errors.append(f"norm-control history missing controlled tensors: {missing_history}")
    if mode == "delayed_captured_constant":
        missing_capture = sorted(set(controlled) - set(captured_by_name))
        if missing_capture:
            errors.append(f"delayed norm-control missing capture rows for: {missing_capture}")
        if not targets_path.exists():
            errors.append(f"missing delayed norm-control targets file: {targets_path}")
        else:
            targets = json.loads(targets_path.read_text())
            target_by_name = {record.get("name"): record for record in targets}
            for name, target in captured_by_name.items():
                record = target_by_name.get(name)
                if record is None:
                    errors.append(f"norm_control_targets.json missing {name}")
                elif not close_enough(record.get("target_rms"), target, rtol=rms_rtol):
                    errors.append(f"norm_control_targets.json target mismatch for {name}")
    if projected_rows == 0:
        errors.append("norm-control history has no projected rows")

    return {
        "norm_control_enabled": True,
        "norm_control_mode": mode,
        "norm_control_history_rows": rows,
        "norm_control_controlled_tensors": len(controlled),
        "norm_control_projected_rows": projected_rows,
        "norm_control_max_relative_error": max_relative_error,
        "norm_control_start_step": start_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--norm-control-rms-rtol", type=float, default=1e-5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "tensor_metadata.json"
    history_path = run_dir / "tensor_norm_history.jsonl"
    update_history_path = run_dir / "adamw_update_norm_history.jsonl"
    errors = []

    if not metadata_path.exists():
        errors.append(f"missing metadata file: {metadata_path}")
    if not history_path.exists():
        errors.append(f"missing history file: {history_path}")
    if not update_history_path.exists():
        errors.append(f"missing AdamW update history file: {update_history_path}")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)

    metadata = json.loads(metadata_path.read_text())
    trainable = {record["name"]: record for record in metadata if record.get("trainable")}
    tensor_history = validate_history_file(history_path, trainable, errors)
    update_history = validate_history_file(
        update_history_path,
        trainable,
        errors,
        prefix="adamw_update_",
        require_optimizer_fields=True,
    )
    activation_probe = validate_activation_probe(run_dir, errors)
    norm_control = validate_norm_control(run_dir, trainable, errors, args.norm_control_rms_rtol)
    tensor_steps = set(tensor_history["distinct_steps"])
    update_steps = set(update_history["distinct_steps"])
    if not update_steps:
        errors.append("AdamW update history has no steps")
    if update_steps and not update_steps.issubset(tensor_steps):
        errors.append(f"AdamW update steps not present in tensor norm history: {sorted(update_steps - tensor_steps)}")
    if 0 in update_steps:
        errors.append("AdamW update history should start after the first optimizer step, not at step 0")

    result = {
        "ok": not errors,
        "errors": errors,
        "metadata_tensors": len(metadata),
        "trainable_tensors": len(trainable),
        "history_tensors": tensor_history["tensors"],
        "history_rows": tensor_history["rows"],
        "distinct_steps": tensor_history["distinct_steps"],
        "min_records_per_tensor": tensor_history["min_records_per_tensor"],
        "max_records_per_tensor": tensor_history["max_records_per_tensor"],
        "adamw_update_history_tensors": update_history["tensors"],
        "adamw_update_history_rows": update_history["rows"],
        "adamw_update_distinct_steps": update_history["distinct_steps"],
        "adamw_update_min_records_per_tensor": update_history["min_records_per_tensor"],
        "adamw_update_max_records_per_tensor": update_history["max_records_per_tensor"],
        **activation_probe,
        **norm_control,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
