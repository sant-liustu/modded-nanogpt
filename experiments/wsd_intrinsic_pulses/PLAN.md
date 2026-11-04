# Intrinsic-time equal-width pulse experiment

Direct implementation requested in chat: w768/d12 norm-controlled AdamW probes.
From stable checkpoints5000/10000/15000/16000 independently fork baseline0.06
for520 updates and pulse ELR0.05/0.04/0.03/0.02/0.01 for24/30/40/60/120
updates (intrinsic width1.2), then0.06 for500 updates (intrinsic recovery30).
Each branch has total intrinsic time31.2. All share fixed validation prefix,
default1048576 validation tokens EVERY update including initial checkpoint.
Global B128/context1024/seed0, same model/optimizer/gamma/projection/data settings.
Checkpoints: pulse-end and final; baseline-final. Full state retained.

Question: amplitude scaling and training-stage dependence in intrinsic time.
Expected observation is a well-measured response, not a claim of LTI. Compare
loss differences at matched intrinsic time, not matched global optimizer step.
Different branch token budgets and Adam/gamma update counts are confounders.
Existing checkpoints must have exact supported source and schedule provenance;
5000/10000/15000 use088c926 w768 source;16000 uses pinned shared-stable source.
We do not assume actual files exist: launcher requires supplied checkpoint paths.
No formal training or cloud execution. No new trainer abstraction: adapt the
existing isolated standalone continuation and add a sequential six-arm launcher.
Files: train_intrinsic_pulse.py, run_probes.py, README, provenance, verification.
Review pulse boundaries, original-stage key on resume, validation, migration
acceptance/rejection, stop after exactly500 recovery updates, state preservation.
