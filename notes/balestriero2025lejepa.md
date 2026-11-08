# Balestriero & LeCun 2025, LeJEPA

**Claim**
LeJEPA trains a JEPA with two terms: a prediction loss between views and SIGReg, a regulariser that pushes the embedding law toward an isotropic Gaussian. The authors argue the isotropic Gaussian is the optimal embedding law for downstream linear and k-NN probes. With it, they drop heuristics such as stop-gradient, EMA teachers and predictor tricks, and they report stable training across architectures.

**Method**
SIGReg projects embeddings onto random unit directions (slices) and applies a 1-D Epps–Pulley test, which compares the empirical characteristic function to a Gaussian one, to each projection. It averages the results. The integral is computed by trapezoidal quadrature, and 17 knots suffice per their ablation. Directions are resampled every step. The number of slices is a tunable hyperparameter (toy figures use M=10 or 16), and the paper reports only a modest sensitivity to it. The authors say the exact integral recovers a kernel MMD per slice. The optimality argument bounds the bias and variance of linear and k-NN probes and shows that, under a scalar-covariance constraint, the isotropic Gaussian uniquely minimises the integrated squared bias. Their Cramér–Wold argument gives a sampling bound on the number of slices needed. I did not check the proofs.

**Overlap with this study**
- Direct baseline: sliced, Gaussian-prior distributional regularisation of JEPA latents.
- Its link to MMD supports our MMD and KSD comparisons.
- It states that a moment-matching T gives a VICReg-like limit. This is the VICReg baseline relationship we need.
- Its optimality claim is the identifiability-adjacent statement we can test.

**Open gap**
- No nuisance model: views are augmentations with no explicit independent nuisance, and nuisance leakage is not measured.
- No explicit dimensional versus information bottleneck, and no injected-noise sweep of σ against d.
- No Stein or KSD regulariser and no IMQ kernel.
- No temporal JEPA or autoencoder setting.
- No ground-truth factors (3DShapes) for identifiability.
- It does not show that Gaussianity helps when nuisance is present.

**BibTeX**
```
@misc{balestriero2025lejepaprovablescalableselfsupervised,
      title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics}, 
      author={Randall Balestriero and Yann LeCun},
      year={2025},
      eprint={2511.08544},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.08544}, 
}
```

**Table row**
- regulariser: SIGReg (sketched Epps–Pulley)
- sliced/full: sliced
- prior: isotropic Gaussian
- nuisance: not treated (augmentation views only)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: not formal (optimal-embedding argument for probes)
- objective: JEPA
- data: natural (ImageNet-1k, Galaxy10 and others)
