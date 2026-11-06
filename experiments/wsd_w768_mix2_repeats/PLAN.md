# w768/d12: additional280 runs =140 paired mixed-two-shard repeats

User asks for many more trajectories and explicitly proposes loading two shards
and shuffling. Replace prior single-shard sequential protocol with a separate
mixed-pool experiment. Keep original shared16000 weights/Adam/schedulers/RNG,
fixed matrix RMS and gamma.0018. Baseline.06*320; pulse.01*120+.06*300.
Fixed same validation prefix every update,default1048576 tokens.

Each pair uses a distinct data seed (default2000..2139). Pick two different shards
from the checkpoint's unused suffix (skip its entire current shard). Can fix the
two shards explicitly. Load uint16 arrays separately, split each into contiguous
1024-token contexts with next-token targets within the same shard, shuffle context
IDs without replacement, form minibatches across the pool. Never shuffle individual
tokens or join arbitrary shard edges. All ranks share one global permutation,
read disjoint row slices. Pair arms share identical chosen shards/seed/permutation
and physical-step batch prefix; different pairs get different data trajectories.
A pool is reused across pairs; these are not140 independent new corpora.

Standalone copy of existing w768 trainer; extend only its existing loader mode and
state. No shared-trainer abstraction. Save both shard IDs,permutation,cursor,next
batch and order fingerprint; only initial historical migration replaces old prefetch.
Same-pair resume never resets data. Accept known legacy shared16k and original
five-model-suite w768 shared16k. Keep strict model/config/runtime checks.

New launcher plans140 pairs/280 runs. Separate fixed-val paired delta mean/SD/SE;
old sequential-shard results are not blindly pooled with new-protocol estimates.
Controls unchanged. Expected outcome is correct data/state and measurements,
not proven LTI or guaranteed variance reduction. Risks: finite-pool overlap,
nonIID shards,unequal data/Adam/gamma steps at matched intrinsic time.

Review:preserve context order within samples,valid next-token labels,no block
crossing shard boundary,pair/rank agreement,140 unique permutations,checkpoint
roundtrip and statistical pairing. RealPython3.10/3.12 stdlib checks and tinyCUDA
loader/complete-state tests. No actual user checkpoint or formal cloud run.

Default bulk runs save metrics/data-plan only;full branch checkpoints optional via --save-checkpoints,or explicit --save-at. Both stop paths tested.
