# Twenty fresh-shard paired repeats from shared stable16000

User correction: no token-block shuffle. Use20 distinct entirely unvisited shards,
one per repeat, sequential tokens from shard start, paired baseline.06*520 and
pulse.01*120 then.06*500. Same16k weights/Adam/gamma/norm targets/model RNG;
fixed validation prefix every update. Replace saved training prefetch initially.

Prove freshness by replaying the pinned original sorted cyclic loader's microbatch
cursor for16000*gradient_accumulation+1 fetches. Match saved current shard and
rank-local position. Conservatively exclude every shard trained/prefetched/loaded,
including partially consumed ones. If earlier cycling touched all shards or saved
cursor cannot be proven, reject. Select20 distinct remaining shards large enough
for620 updates plus one prefetched microbatch. Do not spill or wrap to old shards.
Require exact saved data paths/signatures and known shared-stable source/schedule.

Initial migration uses selected shard from offset0; same-repeat resumes keep saved
shard/cursor/next batch. Pair identity and shard identity are checkpoint metadata.
No new trainer abstraction; small shared stdlib history-audit helper, isolated
standalone trainer, sequential launcher and exact intrinsic-grid aggregation.

Report paired pulse-minus-baseline mean/sample SD/SE on521 exact tau points;
shards may not beIID, intervals descriptive. No fresh-validation resampling.
Tests: source equivalence, cursor replay vs actual loader over boundaries/cycles,
corrupted history/capacity rejection,20 distinct shard plan/40 commands;actual
CUDA sequential batches/rank partition, initial reset and bitwise resume;known
paired statistics;realPython3.10/3.12 static tests. No formal training launched.


## User-requested simplification
Use the known sequential checkpoint current_shard and skip its entire prefix.
Shuffle only the remaining shard indices with --shard-seed(default0);choose20.
No historical microbatch replay, no extra file/header/signature scan, no per-shard
length preflight. Retain metadata-only full-corpus-pass guard, original recovery
checks and runtime no-wrap. Developer verification scripts are never called by launcher.
