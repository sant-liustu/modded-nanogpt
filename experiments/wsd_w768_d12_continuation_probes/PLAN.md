# Revision: shared stable validation trajectory before interventions

Start from the pinned088c926 w768/d12 step15000 checkpoint. Extend matrix
ELR0.06 for1000 updates ONCE, measuring fixed-prefix validation after every
update. Save shared step16000, restore all three arms from that complete state.
Pulse=.006*200 then .06*2000 (end18200); high=.06*2000 and low=.006*2000
(end18000). Gamma LR0.0018 and all original optimizer/data/RNG semantics retained.
Every phase uses the identical validation prefix and token budget (default1Mi tokens).
Separate stage-relative and global intrinsic-time metadata supports pre/post plots.

Question: compare cumulative kernel and pulse forgetting against shared stable
validation trajectory. Shared signal/kernel assumptions remain unproven; gamma,
Adam memory, finite validation precision and unequal tokens at matched intrinsic
time are confounders. Expected result here is correctly generated measurements,
not evidence for a loss decomposition. No full-size training is launched.

Minimal changes in existing isolated continuation runner, launcher, tests and docs;
no new trainer abstraction. Original stable source remains untouched. Original15000
checkpoint enters only stable; shared16000 checkpoint may migrate to each branch
under exact source/config/prefix-schedule checks. Same-branch resumes remain strict.
Review every-step validation, single shared prefix, exact phase boundaries, preserved
state and branch-independent checkpoint inputs. Verify synthetic CUDA migration,
resume equality, full schedules and actual save/stop blocks before delivery.
