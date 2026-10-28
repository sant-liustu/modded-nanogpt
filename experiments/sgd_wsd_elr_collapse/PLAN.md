# SGD WSD ELR comparison

User-approved arms: fixed initial per-matrix RMS, sine RMS with LR multiplied
by s, and sine RMS with LR multiplied by s squared. s(u)=1+0.5 sin(2 pi u).
No uncompensated sine arm. All unique trainable 2D matrices are controlled;
1D gamma remains learnable with identical base WSD LR across arms.

Question: does LR/RMS or LR/RMS squared better preserve loss trajectories?
For an exactly scale-invariant objective, gradients scale inversely with the
weight scale, so directional SGD updates scale as LR/norm squared. This model
is not globally scale invariant (including tied embedding/head, V/O and MLP),
so matching ELR need not yield collapse. Gamma co-adaptation is another limit.

Controls: same seed 0 initialization, FineWeb tape order, width768/depth12/heads6,
context1024, batch4, 301728 updates, peak base LR .01; pure SGD without decay,
momentum, clipping or tangent-gradient removal. All arms run from initialization.
WSD uses old w384 proportions: 4000/81600 warmup and 23200/81600 linear decay.
Sine is indexed by consumed-token progress, not LR-dependent intrinsic time.
At step i use norm s(i/N), update with baseLR*s(i/N)^p, then project to s((i+1)/N).

Implementation: standalone train.py and unchanged model snapshot, three shell
launchers, small CPU/CUDA validation. Reuse existing tape and buffered log
recovery code locally, no framework abstraction. Keep original runners intact.
Log each update; validate and per-tensor diagnostics every200 updates, every20
within200 updates of WSD boundaries; checkpoint every1000, boundaries, final,
and permanent200000. Fixed512 validation contexts in batches32.

Verify schedule algebra, norm bounds, same initialization/data, exact CPU
resume including RNG/log recovery, checkpoint contract rejection, and short
CUDA execution. Small tests do not establish collapse or full768 convergence.
Review compensation scope, WSD fractions, progress indexing and norms first.
