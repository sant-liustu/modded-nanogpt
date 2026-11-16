# Grouped Muon norm-control experiments

Run from the repository root with two GPUs and FineWeb shards in data/fineweb10B.

    python experiments/muon_norm_projection_match/run_cloud.py

To reuse the original stage 01 controller history and run only stages 02 and 03:

    python experiments/muon_norm_projection_match/convert_stage01.py /path/to/norm_control_reference/aggregate_match_history.jsonl norm_grouped.jsonl
    python experiments/muon_norm_projection_match/run_cloud.py --norm-reference norm_grouped.jsonl

Stage 02 matches the norm-control reference using MuonW with weight decay 0.1.
Stage 03 uses norm control with weight decay 0 and the Figure 5(b) Hyperball
reference in experiments/hyperball_grouped_elr_match. Each update solves
separately for the block LR (parameter-count-weighted LR / parameter RMS)
and the embedding LR (embedding LR / embedding parameter RMS). Hyperball
targets use raw update RMS as in Figure 5(b). Only the stage 01 reference
keeps its original 10x embedding LR. Each matcher writes both relative
errors and the two LRs to grouped_match_history.jsonl and checks both
errors against 1e-10. Gamma stays on its existing schedule.
