# arnez2026sigreg

Verified via arXiv (bibtex endpoint, abs page metadata, HTML). Title differs from the one in the task: the real title is "The SIGReg Objective as Variational Free Energy: A Theoretical Active-Inference Account of JEPA World Models". Venue: arXiv preprint only (cs.LG, 15 Jul 2026, 28 pp.); no peer-reviewed venue found. Only the first ~100k of 256k HTML characters were read; the nuisance and dimension answers rest on that portion.

## Claim
The choice of anti-collapse regulariser decides whether a JEPA world-model loss (prediction plus weighted regulariser) is a valid Active Inference variational free energy. VICReg and LogDet are unsafe entropy upper bounds, PairDist a safe lower bound, and SIGReg closes the prior-miscalibration gap. Under a constant-noise encoder and successful SIGReg enforcement the objective is an exact information bottleneck.

## Method
Purely theoretical. Z = f(X) + fixed small isotropic Gaussian noise, so H(Z|X) is constant; the noise is an interpretive device, not a capacity limit and not a tuned σ². Target prior N(0,(c/d)I). SIGReg is sliced (random 1-D projections, Cramér–Wold). Proofs in appendix; algebraic core checked in Lean 4. No experiments, no data ("left to separate work"). Relies on Klindt et al. 2026 for linear identifiability; over-complete case unproven.

## Overlap with this study
Closest on SIGReg vs VICReg under an information-bottleneck reading with Gaussian noise injected into z, and on temporal (action-conditioned) JEPA. Cites identifiability. No KSD, MMD, or Stein methods. Nuisance not analysed; d does not vary in any experiment.

## Open gap
No empirical test: no sweep of d against σ², no nuisance-leakage R², no paired-view nuisance setup, no kernel-based (KSD/MMD) regulariser, no real or rendered data. Their fixed-σ² IB claim could be tested by varying σ² and d independently.

## BibTeX
```bibtex
@misc{arnez2026sigregobjectivevariationalfree,
      title={The SIGReg Objective as Variational Free Energy: A Theoretical Active-Inference Account of JEPA World Models}, 
      author={Fabio Arnez and Alexandra Gomez-Villa},
      year={2026},
      eprint={2607.13612},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.13612}, 
}
```

**Table row**
- regulariser · SIGReg (compared theoretically with VICReg, LogDet, PairDist)
- sliced/full · sliced
- prior · isotropic Gaussian N(0,(c/d)I); Gaussian transition prior, variance σ²
- nuisance · — (not treated)
- dimensional bottleneck · — (d only in formulas)
- information bottleneck · yes, theoretical: fixed isotropic Gaussian noise in z gives exact IB under SIGReg
- identifiability · assumed from Klindt et al. 2026 (linear); over-complete unproven
- objective · temporal action-conditioned JEPA
- data · — (no experiments)
