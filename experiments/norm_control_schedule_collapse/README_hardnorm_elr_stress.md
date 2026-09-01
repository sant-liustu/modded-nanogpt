# Hard-norm ELR-govern stress tests

This set contains two matched pairs of Adam experiments.  The controlled set
\(\mathcal C\) is the 72 block Linear matrices plus the tied
`transformer.wte.weight` (73 tensors total).  RMSNorm gamma parameters are not
in \(\mathcal C\): they keep the common WD=0 WSD schedule used by the parent
Adam experiment.

For every \(i\in\mathcal C\), let \(s_i^0=\operatorname{RMS}(W_i(0))\).
The controller samples one schedule once from a recorded seed and assigns it
to each tensor:

\[
q_{\mathrm{constant}}(x)=1,\qquad
q_{\mathrm{up}}(x)=1+x,\qquad
q_{\mathrm{down}}(x)=1-0.5x,
\]
\[
q_{\mathrm{cos}}(x)=1+0.5\sin(2\pi x),\qquad x\in[0,1].
\]

Thus `linear_up` ends at \(2s_i^0\), `linear_down` ends at
\(0.5s_i^0\), and `cosine_cycle` completes exactly one up/down cycle.  State
0 is \(x=0\); after update \(s\), the hard projection uses
\(x=s/T\), where \(T=20400\).

At update \(s\), before Adam, the raw learning rate is set from the shared
ELR target file:

\[
\eta_i(s)=r_i^\star(s)\operatorname{RMS}(W_i(s)).
\]

After the Adam update \(\widetilde W_i\), the parameter is projected to its
assigned trajectory:

\[
W_i(s)=
\frac{s_i^0 q_{g_i}(s/T)}
     {\operatorname{RMS}(\widetilde W_i)}\widetilde W_i.
\]

All controlled groups use fused `AdamW` with `weight_decay=0`, i.e. the
Adam-equivalent update.  The only within-pair difference is the seeded map
\(g_i\); data, model initialization, optimizer, and ELR target are otherwise
the same.

| Experiment | A | B | Shared ELR target |
| --- | --- | --- | --- |
| 1: scalar degeneration | `train_gpt2_gamma_adam_hardnorm_singleelr_wsd005_assignmentA_muonhinit_B0128_devB064.py` (seed 20260901) | `train_gpt2_gamma_adam_hardnorm_singleelr_wsd005_assignmentB_muonhinit_B0128_devB064.py` (seed 20260902) | `rmselr_single_wsd_peak005_B0128_20400.jsonl.gz`: \(r_i^\star(t)=0.05u_{\rm WSD}(t)\) |
| 2: heterogeneous profile | `train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_assignmentA_muonhinit_B0128_devB064.py` (seed 20260903) | `train_gpt2_gamma_adam_hardnorm_pertensor_rmselr_assignmentB_muonhinit_B0128_devB064.py` (seed 20260904) | `rmselr_mixed_attncos_mlpwsd_peak005_007_B0128_20400.jsonl.gz` |

The default arguments of all four scripts point to their own assignment file
and correct target file, so a normal launch needs no ELR/norm-control override.
To regenerate the checked-in target and assignment JSON files:

```powershell
python experiments/norm_control_schedule_collapse/build_hardnorm_stress_inputs.py
```
