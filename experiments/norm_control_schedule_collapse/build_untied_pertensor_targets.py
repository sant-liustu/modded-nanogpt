"""Extend the existing heterogeneous ELR profile to an independent output head."""

import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "rmselr_mixed_attncos_mlpwsd_peak005_007_B0128_20400.jsonl.gz"
DESTINATION = HERE / "rmselr_mixed_attncos_mlpwsd_peak005_007_untied_B0128_20400.jsonl.gz"


def main():
    count = 0
    with gzip.open(SOURCE, "rt", encoding="utf-8") as source:
        with gzip.GzipFile(filename=str(DESTINATION), mode="wb", mtime=0) as target:
            for line in source:
                record = json.loads(line)
                field = ("target_lr_over_tensor_rms" if "target_lr_over_tensor_rms" in record
                         else "target_lr_over_tensor_norms")
                values = record[field]
                assert len(values) == 73 and "lm_head.weight" not in values
                values["lm_head.weight"] = values["transformer.wte.weight"]
                record["untied_head_elr_source"] = "transformer.wte.weight"
                target.write((json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8"))
                count += 1
    assert count == 20400, count
    print(f"Generated {count} rows, 74 tensors: {DESTINATION.name}")


if __name__ == "__main__":
    main()
