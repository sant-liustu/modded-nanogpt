"""Run from the repository root with torchrun."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--reference", help='Omit to record the norm-control reference')
parser.add_argument("--output", required=True)
parser.add_argument("--config", help="JSON hyperparameter overrides")
parser.add_argument("--norm-control-config", default=str(Path(__file__).with_name("projection.json")))
args = parser.parse_args()
settings = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
settings.setdefault("weight_decay", 0.0)
settings["norm_control_config"] = args.norm_control_config if not args.reference else ""
destination = Path(args.output)
rank = int(os.environ.get("RANK", "0"))
if rank == 0 and destination.exists():
    raise FileExistsError(f"Choose a fresh output directory: {destination}")
destination.parent.mkdir(parents=True, exist_ok=True)
# Each rank writes a unique identical config, avoiding process-group initialization here.
config_path = destination.parent / (destination.name + ".config.rank" + str(rank) + ".json")
config_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
sys.argv = [str(Path(__file__).with_name("train.py")), "--output", str(destination), "--config", str(config_path)]
if args.reference:
    sys.argv += ["--reference", args.reference]
else:
    sys.argv += ["--record-reference"]
import runpy
runpy.run_path(sys.argv[0], run_name="__main__")
