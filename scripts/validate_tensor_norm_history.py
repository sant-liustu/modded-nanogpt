import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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
        return {
            f"{prefix}fro_norm",
            f"{prefix}rms_norm",
            f"{prefix}spectral_norm_estimate",
            f"{prefix}spectral_norm_estimate_method",
            f"{prefix}spectral_norm_estimate_block_size",
            f"{prefix}spectral_norm_estimate_iters",
        }
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
        missing = sorted(required_norm_fields(ndim, prefix) - set(record))
        if missing:
            errors.append(f"{path.name} line {line_number}: missing {missing} for {name}")
        if ndim == 2:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
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
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
