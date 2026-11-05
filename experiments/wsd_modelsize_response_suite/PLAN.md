# Five-size response suite

Direct user request: extend prior response scripts to classic five sizes w384/d12,
w768/d6,d12,d24,w1536/d12. Fixed-RMS norm control,shared ELR peak.06,
gamma LR.0018,globalB128/context1024 and exact historical optimizer/data state.

1.Time dependence: checkpoints5000/10000/15000/16000,baseline.06*520,
pulse.01*120+.06*500 and.03*40+.06*500.
2.Amplitude: ONLY16000,baseline and pulses.05*24,.04*30,.03*40,.02*60,.01*120,
each followed by500 updates at.06. Shared historical16000 arms reused across1/2.
3.Noise repeats:16000,20 pairs per selected model,baseline/.01 pulse,distinct
remaining shards after current shard,shuffle shard list only,fixed shard seed.
No token shuffle,no heavy history replay,no per-shard length preflight.
Time/amplitude probes preserve the saved next batch; fresh pairs replace it.

If no16k checkpoint exists,provide stable continuation from15k to16k at.06;
never run original trainer beyond15k because its planned decay starts there.
Preparation checkpoint is shared across all16000 branches. Its val is measured
every update. Existing pinned w768/d12 shared16k inputs remain supported.

All probes use the SAME fixed val prefix EVERY update,default1,048,576 tokens.
Common intrinsic grid baseline i vs pulse6i/.01,2i/.03,etc; paired averages from20
fresh shards retain original exact521-point baseline/.01 grid. Model sizes have
separate checkpoints and original rank/microbatch settings:infer from checkpoint.
No checkpoint path is guessed or actual cloud training started.

Minimal implementation:five isolated standalone copies of existing trainer,one
simple planning/runner script plus existing aggregator. No shared trainer refactor.
Known original source hashes per model and common schedule hash explicitly pinned.
Same-suite stable16k may migrate into all branches;same-branch resume preserves
mode,arm,original fork,fresh replicate/shard seed and next batch. Rejectwrongmodel.

Verification:realPython3.10/3.12 stdlib source/schedule/plan checks;5-model synthetic
CUDA migration/resume with allmodel route identities and negative tests;actual
meta model parameter counts;launcher dependency/scope/dedup;fresh data rank/stream
and paired statistics inherited checks. These are local checks,not formal-size
DDP/compile or real user checkpoint evidence. User review focus:scope counts,
original model identity,stable16k preparation,mode/data and checkpoint boundaries.
