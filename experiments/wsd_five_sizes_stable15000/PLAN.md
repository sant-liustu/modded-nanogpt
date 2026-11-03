# Five-size WSD stable checkpoint experiment

Question: obtain matched-history checkpoints for later decay-response experiments.
Five sizes: w384/d12, w768/d6,d12,d24, w1536/d12. Fixed global B128,
context1024, seed0, fan-in initialization, AdamW, matrix/gamma WD0 and
per-update projection to initial matrix RMS. Same input shards/order required.
Matrix ELR: warmup924 updates to0.06; constant through15000; nominal linear
decay3858 updates to zero at18858. Gamma absolute LR0.0018 warms up924
updates then remains constant. Default execution stops at15000.
This changes matrix warmup from the old linear suite's1886 to924, matching
the previously discussed proportional WSD warmup. It is not an identical-prefix
comparison with existing linear runs.

Save complete rank-local checkpoints at5000,10000,15000: model, optimizer,
scheduler, fixed norm targets, RNG, loader positions and prefetched next batch.
Expected observations: acquisition of checkpoints only; no scientific outcome
is established by implementation checks. Later identical-current-ELR probes
can distinguish fast response from history dependence, without presupposing
that residual loss is accumulated noise.

Implementation: five isolated standalone trainer copies and existing launcher;
no new trainer abstraction. Only schedule, checkpoint/stop controls and metadata
change. Old source directories remain unchanged. Scientific settings remain
locked. Validation stays disabled as in the source suite; loss/diagnostics are
preserved. Risks: full-size GPU capacity unmeasured, large checkpoint disk use,
data ordering must match, gamma has its own dynamics. Resume remains strict;
arbitrary future decay forks require a separately reviewed continuation change.
Review: shared_elr, warmup924, default stop15000, checkpoint completeness,
unchanged model/optimizer/projection/data loader functions and launcher mapping.
