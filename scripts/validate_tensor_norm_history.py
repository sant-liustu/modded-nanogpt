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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metadata_path = run_dir / "tensor_metadata.json"
    history_path = run_dir / "tensor_norm_history.jsonl"
    errors = []

    if not metadata_path.exists():
        errors.append(f"missing metadata file: {metadata_path}")
    if not history_path.exists():
        errors.append(f"missing history file: {history_path}")
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        raise SystemExit(1)

    metadata = json.loads(metadata_path.read_text())
    trainable = {record["name"]: record for record in metadata if record.get("trainable")}
    seen_by_tensor = defaultdict(list)

    for line_number, record in load_jsonl(history_path):
        name = record.get("name")
        if name not in trainable:
            errors.append(f"line {line_number}: unknown tensor {name}")
            continue
        ndim = trainable[name]["ndim"]
        if record.get("ndim") != ndim:
            errors.append(f"line {line_number}: ndim mismatch for {name}")
        for key, value in record.items():
            if key.endswith("_norm") or key == "abs_value":
                if not isinstance(value, (int, float)):
                    errors.append(f"line {line_number}: non-numeric {key} for {name}")
                elif value != value or value in (float("inf"), float("-inf")):
                    errors.append(f"line {line_number}: non-finite {key} for {name}")
        if ndim == 2:
            required = {"fro_norm", "rms_norm", "spectral_norm"}
        elif ndim == 1:
            required = {"fro_norm", "rms_norm"}
        elif ndim == 0:
            required = {"abs_value", "rms_norm"}
        else:
            required = {"fro_norm", "rms_norm"}
        missing = sorted(required - set(record))
        if missing:
            errors.append(f"line {line_number}: missing {missing} for {name}")
        seen_by_tensor[name].append(record.get("step"))

    missing_tensors = sorted(set(trainable) - set(seen_by_tensor))
    if missing_tensors:
        errors.append(f"missing history for tensors: {missing_tensors}")

    counts = Counter(len(steps) for steps in seen_by_tensor.values())
    distinct_steps = sorted({step for steps in seen_by_tensor.values() for step in steps})
    result = {
        "ok": not errors,
        "errors": errors,
        "metadata_tensors": len(metadata),
        "trainable_tensors": len(trainable),
        "history_tensors": len(seen_by_tensor),
        "history_rows": sum(len(steps) for steps in seen_by_tensor.values()),
        "distinct_steps": distinct_steps,
        "min_records_per_tensor": min(counts) if counts else 0,
        "max_records_per_tensor": max(counts) if counts else 0,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
