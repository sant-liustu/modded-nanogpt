# Distributed Fisher validation, 2026-10-28

Production: two-rank DDP training and two-rank Fisher. Each independent draw
partitions GLOBAL 1024 sequences across ranks: 512 each, micro-batch64 and
8 local accumulations. Local losses divide by GLOBAL valid-token count. Private
gradient buffers SUM-all-reduce before squaring; no averaging local Fisher.
Two draws have independent rank-aware seeds. Rank 0 logs means, SE and RMS.

PASS: verify_fisher_distributed.py, actual two-process CPU/Gloo, small linear
softmax model and masked inputs. Both distributed draws match a serial complete
objective using the exact same sampled labels for each rank/micro-batch. Global
token scaling, sequence counts, seeds, identical rank outputs, standard error,
unchanged gradients/RNG/modes all checked. TCP uses use_libuv=0 for the installed
Windows PyTorch. Test required local loopback access outside sandbox restrictions.

PASS: verify_fisher_accumulation.py, existing single-process formula/schedule tests.
PASS: real FineWeb single-CUDA eager w128/d1/T8 four-update off/on and resume checks:
- local_checks/fisher_train_20261028_192812/report.json
- local_checks/fisher_train_muon_20261028_192839/report.json

No local two-GPU hardware; NCCL, two-GPU compiled training, full-length memory
and speedup have not been measured. Two-process CPU correctness does not establish
these performance/runtime properties. Training optimizer/schedule/norm control
settings are unchanged. New source/config still requires compatible checkpoints.
