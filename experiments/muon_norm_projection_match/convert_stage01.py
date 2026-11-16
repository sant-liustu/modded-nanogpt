"""Recover the two norm-control targets from the old stage 01 controller log."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


def convert(history, output, steps=20400):
    metadata = Path(str(output) + ".metadata.json")
    if output.exists() or metadata.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "reference.jsonl"
        count = 0
        with history.open(encoding="utf-8") as src, staged.open("w", encoding="utf-8") as dst:
            for count, line in enumerate(src, 1):
                row = json.loads(line)
                if count == 1:
                    names, sizes = row["names"], row["numels"]
                    emb = names.index("transformer.wte.weight")
                    block_size = sum(sizes) - sizes[emb]
                rms = row["rms"]
                if (row["update_step"] != count or row["pre_update_step"] != count-1
                        or row["weight_decay"] != 0 or row["names"] != names
                        or row["numels"] != sizes or len(rms) != len(sizes)
                        or not all(math.isfinite(x) and x > 0 for x in rms)
                        or not all(math.isfinite(row[k]) and row[k] > 0 for k in ("base_lr","embed_lr"))):
                    raise ValueError(f"Invalid stage 01 row at step {count}")
                block = row["base_lr"] * math.fsum(n/r for i,(n,r) in enumerate(zip(sizes,rms)) if i != emb) / block_size
                dst.write(json.dumps(dict(update_step=count, pre_update_step=count-1,
                    block_target=block, embedding_target=row["embed_lr"]/rms[emb])) + "\n")
        if count != steps:
            raise ValueError(f"Expected {steps} steps, found {count}")
        info = Path(tmp) / "meta.json"
        info.write_text(json.dumps(dict(metric="block_weighted_and_embedding_parameter_rms",
            names=names, numels=sizes, total_controlled_numel=sum(sizes))), encoding="utf-8")
        staged.rename(output)
        info.rename(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.history, args.output)
