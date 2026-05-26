import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_jsonl(path, errors):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name} line {line_number}: invalid JSON: {exc}")
    return rows


def validate_metadata(path, errors):
    if not path.exists():
        errors.append(f"missing metadata file: {path}")
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid metadata JSON: {exc}")
        return {}
    if not isinstance(records, list):
        errors.append("metadata must be a JSON array")
        return {}

    by_name = {}
    for index, record in enumerate(records):
        name = record.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"metadata row {index}: missing name")
            continue
        by_name[name] = record
        shape = record.get("shape")
        if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            errors.append(f"metadata row {index}: invalid shape for {name}")
        ndim = record.get("ndim")
        if not isinstance(ndim, int) or ndim != len(shape or []):
            errors.append(f"metadata row {index}: invalid ndim for {name}")
        numel = record.get("numel")
        if not isinstance(numel, int) or numel <= 0:
            errors.append(f"metadata row {index}: invalid numel for {name}")
        if not isinstance(record.get("dtype"), str):
            errors.append(f"metadata row {index}: missing dtype for {name}")
        if not isinstance(record.get("trainable"), bool):
            errors.append(f"metadata row {index}: missing trainable for {name}")
    return by_name


def validate_norm_fields(record, prefix, errors, context):
    for key in (f"{prefix}fro_norm", f"{prefix}rms_norm"):
        if not is_finite_number(record.get(key)):
            errors.append(f"{context}: missing or non-finite {key}")


def validate_tensor_history(path, trainable, errors):
    if not path.exists():
        errors.append(f"missing tensor history file: {path}")
        return {"rows": 0, "steps": [], "tensors": []}
    rows = read_jsonl(path, errors)
    seen = defaultdict(set)
    trainable_names = {name for name, record in trainable.items() if record.get("trainable")}
    for index, record in enumerate(rows):
        context = f"{path.name} row {index}"
        name = record.get("name")
        if name not in trainable_names:
            errors.append(f"{context}: unknown or non-trainable tensor {name}")
            continue
        step = record.get("step")
        if not isinstance(step, int) or step < 0:
            errors.append(f"{context}: invalid step")
        else:
            seen[step].add(name)
        metadata = trainable[name]
        if record.get("shape") != metadata.get("shape"):
            errors.append(f"{context}: shape mismatch for {name}")
        if record.get("ndim") != metadata.get("ndim"):
            errors.append(f"{context}: ndim mismatch for {name}")
        validate_norm_fields(record, "", errors, context)

    for step, names in seen.items():
        missing = sorted(trainable_names - names)
        if missing:
            errors.append(f"{path.name}: step {step} missing tensor rows: {missing[:5]}")
    return {
        "rows": len(rows),
        "steps": sorted(seen),
        "tensors": sorted({record.get("name") for record in rows if isinstance(record.get("name"), str)}),
    }


def validate_update_history(path, trainable, errors):
    if not path.exists():
        errors.append(f"missing optimizer update history file: {path}")
        return {"rows": 0, "steps": [], "optimizer_types": []}
    rows = read_jsonl(path, errors)
    seen = defaultdict(set)
    optimizer_types = Counter()
    trainable_names = {name for name, record in trainable.items() if record.get("trainable")}
    for index, record in enumerate(rows):
        context = f"{path.name} row {index}"
        name = record.get("name")
        if name not in trainable_names:
            errors.append(f"{context}: unknown or non-trainable tensor {name}")
            continue
        step = record.get("step")
        if not isinstance(step, int) or step <= 0:
            errors.append(f"{context}: invalid update step")
        else:
            seen[step].add(name)
        optimizer_type = record.get("optimizer_type")
        if optimizer_type not in {"AdamW", "Muon"}:
            errors.append(f"{context}: invalid optimizer_type {optimizer_type}")
        else:
            optimizer_types[optimizer_type] += 1
        for key in ("lr", "weight_decay"):
            if not is_finite_number(record.get(key)):
                errors.append(f"{context}: missing or non-finite {key}")
        for key in ("optimizer_index", "param_group_index"):
            if not isinstance(record.get(key), int):
                errors.append(f"{context}: missing integer {key}")
        metadata = trainable[name]
        if record.get("shape") != metadata.get("shape"):
            errors.append(f"{context}: shape mismatch for {name}")
        if record.get("ndim") != metadata.get("ndim"):
            errors.append(f"{context}: ndim mismatch for {name}")
        validate_norm_fields(record, "param_before_", errors, context)
        validate_norm_fields(record, "param_after_", errors, context)
        validate_norm_fields(record, "applied_update_", errors, context)

    for step, names in seen.items():
        missing = sorted(trainable_names - names)
        if missing:
            errors.append(f"{path.name}: step {step} missing update rows: {missing[:5]}")
    for required_type in ("AdamW", "Muon"):
        if optimizer_types[required_type] == 0:
            errors.append(f"{path.name}: missing optimizer type {required_type}")
    return {
        "rows": len(rows),
        "steps": sorted(seen),
        "optimizer_types": sorted(optimizer_types),
    }


def default_log_file_for_run(run_dir):
    return run_dir.parent / f"{run_dir.name}.txt"


def validate_ema_loss_log(path, ema_names, errors):
    if not path.exists():
        errors.append(f"missing text log file for EMA loss check: {path}")
        return {"log_file": str(path), "ema_fields": []}
    text = path.read_text(encoding="utf-8", errors="replace")
    required_fields = ["val_loss/raw", *[f"val_loss/{name}" for name in ema_names]]
    found_fields = []
    for field in required_fields:
        if f"{field}:" not in text:
            errors.append(f"{path.name}: missing EMA validation field {field}")
        else:
            found_fields.append(field)
    return {"log_file": str(path), "ema_fields": found_fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--require-ema-losses", action="store_true")
    parser.add_argument("--ema-names", default="ema_h32,ema_h128")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    errors = []
    metadata = validate_metadata(run_dir / "tensor_metadata.json", errors)
    tensor_history = validate_tensor_history(run_dir / "tensor_norm_history.jsonl", metadata, errors)
    update_history = validate_update_history(run_dir / "optimizer_update_norm_history.jsonl", metadata, errors)
    ema_loss_log = None
    if args.require_ema_losses:
        log_file = Path(args.log_file) if args.log_file else default_log_file_for_run(run_dir)
        ema_names = [name.strip() for name in args.ema_names.split(",") if name.strip()]
        ema_loss_log = validate_ema_loss_log(log_file, ema_names, errors)

    result = {
        "ok": not errors,
        "errors": errors,
        "metadata_tensors": len(metadata),
        "trainable_tensors": sum(1 for record in metadata.values() if record.get("trainable")),
        "tensor_history_rows": tensor_history["rows"],
        "tensor_history_steps": tensor_history["steps"],
        "optimizer_update_rows": update_history["rows"],
        "optimizer_update_steps": update_history["steps"],
        "optimizer_types": update_history["optimizer_types"],
    }
    if ema_loss_log is not None:
        result["ema_loss_log"] = ema_loss_log
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
