"""Build the grouped stage 01 reference on the cloud and run stages 02 and 03."""
import argparse
from pathlib import Path
import subprocess
import sys
import uuid

from convert_stage01 import convert

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, help="Original stage 01 aggregate_match_history.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Build the reference and print both commands without training")
    args = parser.parse_args()
    history = args.history
    if history is None:
        candidates = sorted((REPO / "logs").glob("muon_projection_*/norm_control_reference/aggregate_match_history.jsonl"))
        if len(candidates) != 1:
            parser.error("Expected one old stage 01 history. Specify --history PATH. Found: "
                         + (", ".join(map(str, candidates)) or "none"))
        history = candidates[0]
    history = history.resolve()
    if not history.is_file():
        parser.error(f"Missing stage 01 controller history: {history}")
    reference = REPO / "logs" / "grouped_references" / uuid.uuid4().hex / "reference.jsonl"
    print(f"Stage 01 input: {history}", flush=True)
    convert(history, reference)
    print(f"Built 20,400 grouped targets: {reference}", flush=True)
    command = [sys.executable, str(HERE / "run_cloud.py"), "--norm-reference", str(reference)]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
