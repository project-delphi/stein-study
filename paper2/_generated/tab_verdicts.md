| # | prediction | verdict | key statistic |
| :--- | :--- | :--- | :--- |
| P1 | step-like onset | refuted (cov, VICReg); step holds for kernel arms | kernel arms rise within 0.30–0.60 decades; cov, VICReg ≥ 0.10 at lowest λ |
| P2 | location within ×3 | refuted | KSD 0.32× [0.29, 0.37], sliced KSD 0.29× [0.22, 0.38], SIGReg 0.72×, MMD 0.79× |
| P3 | ordering SIGReg ≈ cov < KSD < sKSD < MMD | confirmed (cov censored) | successive ratios 10.5, 7.3, 6.3; CIs exclude 1 |
| P4 | α ratio in [α(2−α), 1] | confirmed at α = 0.25 (4 kernel arms); 2 above 1 at α = 0.5 | ratio at α = 0.25: 0.50–0.65 (∝ α: 0.25) |
| P5 | no threshold at d = 6 | refuted | all 6 arms reach ≥ 0.10 at d = 6 (max 0.33–0.67) |
| P6 | d = 24 ratios | not refuted; point values miss | KSD 0.36, SIGReg 0.76, MMD 0.81 |
| P7 | probe steeper than share | inconclusive | probe narrower in 3/6 arms |
| P8 | linear rung within ×3 | refuted (cov location and α ratio) | cov 9.4×, KSD 1.33× theory; α ratio cov 0.69, KSD 1.06 |
| P9 | cov form 1 − λ*/λ | cov not refuted (one range met per rung); kernel part confirmed | linear-rung cov v = 0.30 / 0.53 at 2/5 λ̂* |
| P10 | KSD/cov > 5 | refuted (linear rung); consistent (conv) | linear rung 3.1; conv > 14.8 (cov censored) |
| P11 | leakage persists at 8000 steps | refuted | 8000 − 2000 steps: −0.36, −0.41, −0.33, −0.18 |
| P12 | spare directions (descr.) | – | unexplained share rises; kNN content R² → 1.00 |
| P13 | sign of batch σ (KSD) | confirmed (leakage) | KSD batch σ = 1: +0.66, +0.65 |
| P14 | size of σ effect (KSD) | refuted in size | λ = 0.2, σ = 1: +0.65 (predicted 0) |
| P15 | fixed vs batch σ | refuted (KSD) | fixed − batch at σ 0.5: +0.13, +0.15 |
| P16 | no orth-fit $R^2$ gain from matched prior | not refuted | uniform: −0.02, +0.03 (CIs include 0) |
| P17 | no hue-pair MCC gain | refuted | hue-pair MCC +0.25 at uniform λ 0.2 |
| P18 | top-k cov arm | not tested | E6 cut |
