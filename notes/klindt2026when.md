# Klindt, LeCun, Balestriero 2026: When Does LeJEPA Learn a World Model?

Status: VERIFIED (arXiv record and arXiv HTML full text read; arXiv listing only, no venue confirmed; stat.ML, v1 25 May 2026).

**Claim**
LeJEPA (alignment plus SIGReg toward an isotropic Gaussian) recovers the latents up to an orthogonal map, and within a specific world class this holds only for Gaussian latents. The reported claim is confirmed, but the converse is conditional on that class (independent, stationary, additive-noise transitions), not on all latent laws.

**Method**
Theorem 1: for z ~ N(0, I_n) with OU transition z' = rho z + sqrt(1-rho^2) eta, any measurable h with h(z) ~ N(0, I_n) has alignment loss >= 2(1-rho)n, with equality iff h(z) = Qz, Q orthogonal (Hermite/Mehler spectral argument). Theorem 2: if every whitened minimiser is linear, z is Gaussian (affine eigenfunction forces a linear score). Theorem 3: approximate bound from alignment gap and whitening error. Population-level; SIGReg assumed achieved; output dimension m = n (m != n left open); no nuisance variable, since views differ only by transition noise.
Experiments: 2D mixings; N up to 1024 comparing SIGReg, VICReg, InfoNCE; a generalized-normal sweep (Laplace to uniform; R^2 peaks at Gaussian); DMC Reacher pixels (OU versus RL trajectories, R^2 below 0.5 on trajectories).

**Overlap with this study**
Gaussian-prior JEPA identifiability, and their sweep resembles our uniform/Laplace content laws; their theory predicts a Gaussian-targeting regulariser fails for non-Gaussian content. No KSD, MMD or EMA/stop-grad; no 3DShapes/CIFAR/Moving-MNIST.

**Open gap**
Nuisance in views, d != n, injected-noise (information) bottlenecks, KSD/MMD regularisers, matched non-Gaussian priors (they vary the data law, not the prior), real data with known factors.

**BibTeX**
```
@misc{klindt2026doeslejepalearnworld,
      title={When Does LeJEPA Learn a World Model?}, 
      author={David Klindt and Yann LeCun and Randall Balestriero},
      year={2026},
      eprint={2605.26379},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.26379}, 
}
```

**Table row**
- regulariser: SIGReg (also VICReg, InfoNCE compared)
- sliced/full: sliced (SIGReg sketch)
- prior: isotropic Gaussian (non-Gaussian latent laws tested as the data law, not as priors)
- nuisance (treated? how): not treated; the views differ only by OU transition noise
- dimensional bottleneck: not studied; assumes m = n (flagged as open)
- information bottleneck: —
- identifiability: linear up to orthogonal Q; iff Gaussian within the stated world class; approximate bound
- objective: JEPA (LeJEPA; temporal/stationary pairs)
- data: synthetic (2D, up to 1024-d) and rendered (DMC Reacher pixels)
