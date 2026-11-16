"""Run both experiments with fixed formal configurations on two GPUs."""
import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-reference", type=Path, help="Existing grouped norm-control reference; skip stage 01")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = REPO / "data/fineweb10B"
    hyperball = HERE.parent / "hyperball_grouped_elr_match/hyperball_grouped_reference.jsonl"
    for path in [hyperball, Path(str(hyperball) + ".metadata.json"), HERE / "projection.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.dry_run:
        for pattern in ["fineweb_train_*.bin", "fineweb_val_*.bin"]:
            if not glob.glob(str(data / pattern)):
                raise FileNotFoundError(f"Missing dataset shards: {data / pattern}")
    output = REPO / "logs" / ("muon_projection_" + uuid.uuid4().hex)
    projection = str(HERE / "projection.json")
    stages = [
        ("norm_control_reference", 0.0, projection, None),
        ("muonw_match_norm_control", 0.1, "", output / "norm_control_reference/reference.jsonl"),
        ("norm_control_match_muonh", 0.0, projection, hyperball),
    ]
    if args.norm_reference:
        ref = args.norm_reference.resolve()
        meta = json.loads(Path(str(ref) + ".metadata.json").read_text(encoding="utf-8"))
        if not ref.is_file() or meta["metric"] != "block_weighted_and_embedding_parameter_rms":
            raise ValueError("Expected grouped norm-control reference")
        stages = [("muonw_match_norm_control", 0.1, "", ref), stages[2]]
    env = os.environ.copy()
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "TORCHELASTIC_RUN_ID"]:
        env.pop(key, None)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["PYTHONUNBUFFERED"] = "1"
    if len(env["CUDA_VISIBLE_DEVICES"].split(",")) != 2:
        raise ValueError("Two visible GPUs are required")
    for name, weight_decay, norm_control, reference in stages:
        config = dict(input_bin=str(data / "fineweb_train_*.bin"), input_val_bin=str(data / "fineweb_val_*.bin"), batch_size=128, device_batch_size=64, num_iterations=20400, seed=0, weight_decay=weight_decay, norm_control_config=norm_control)
        config_path = output / (name + ".json")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2", "--max-restarts=0", str(HERE / "train.py"), "--output", str(output / name), "--config", str(config_path)]
        command += ["--reference", str(reference)] if reference else ["--record-reference"]
        print(json.dumps(dict(stage=name, config=config, command=command), indent=2), flush=True)
        if args.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
