# Cosine-wave frequency sweep

## Experiment question

Test whether gamma norm-control collapse remains accurate as the norm/LR
schedule oscillates more rapidly.

## Controlled comparison

All scripts are direct copies of
`train_gpt2_gamma_cosine_wave_lrmatched_B0128_devB064.py`. They preserve the
learnable gamma architecture, optimizer groups, norm-control targets, amplitude
`0.5`, phase, delayed start at step 1,000, LR/norm matching, WSD envelope,
20,400 updates, monitoring, and seed.

Only `period_steps` changes:

| nominal waves | period steps | cycles after control start |
| ---: | ---: | ---: |
| 2 (original) | 10,000 | 1.94 |
| 4 | 5,000 | 3.88 |
| 8 | 2,500 | 7.76 |
| 16 | 1,250 | 15.52 |
| 32 | 625 | 31.04 |

The nominal labels follow the original convention of calling the 10,000-step
period script a two-wave experiment over a roughly 20k-step run.

## Verification

1. Verify the four copies differ from the original only in header text and
   `period_steps`.
2. Count peaks/troughs and confirm the common amplitude and phase.
3. Generate the exact block-LR schedule table and comparison figure.
4. Run short CUDA smoke tests with a scaled control horizon so every variant
   executes multiple oscillations.
