# Two smaller models:400 paired pulse repeats per model

Direct request:extend the latest w768/d12 mixed-two-shard protocol to w384/d12
and w768/d6,400 pairs EACH (800 runs/model,1600 total). Same shared16000 state;
baseline.06*270; pulse.01*120 then.06*250; equal total intrinsic time16.2.
Model-specific architecture,original optimizer/gamma/norm control/data/RNG retained.
Each seed4000..4399 selects two unused-suffix shards and shuffles intact contexts;
paired arms share exactly the same pool/order. Fixed val prefix every update,
default1048576 tokens. Default metrics-only,optional full checkpoints.

No new trainer abstraction:two isolated standalone copies of validated mixed
trainer. Parent shared16000 source SHA pinned separately per architecture. Include
byte-identical corresponding stable preparation scripts for missing16000 inputs.
A single launcher can select one model or run both via JSON;all scientific/state
checks remain. No historical replay or selected-shard length preflight beyond
required pooled context capacity. No formal training or cloud command launched.

Review:400 pairs versus400 runs;model/parent source binding and actual parameter
counts;stable16k preparation;paired data identity;250 recovery,270 baseline,370
pulse stops;Python3.10.16 cloud vs3.12 local compatibility;do not pool models.
Tests:real3.10/3.12 source/plan/400 permutation/statistics checks;actual small
architectures short eager CUDA optimizer/projection checks;representative complete
mixed checkpoint migration/resume plus rank/label validation and wrong-model rejection.
Synthetic checks are not full B128/T1024/DDP/compile or real user checkpoint evidence.
