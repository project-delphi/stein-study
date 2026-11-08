# Nicollier 2026 — Beyond Gaussian Worlds

Status: VERIFIED (title, authors, year, arXiv ID from arXiv BibTeX and abs page; arXiv preprint marked "Preprint", no venue; the export API was rate-limited, and the method and experiment details come from the PDF text).

**Claim**
Klindt et al. showed that Gaussian targets give linear identifiability for Euclidean worlds, and only Gaussians do. This paper extends that result to latent states on embedded Riemannian manifolds. Uniform-on-sphere latents matched to a uniform-sphere target are recovered up to an orthogonal map, so Gaussian uniqueness is not universal. The paper also gives a tighter approximate-recovery bound for the sphere. Matching the target to the latent geometry gives the best linear recovery on Gaussian, sphere, torus and Clifford-torus worlds.

**Method**
- Objective: alignment (invariance) loss plus a distribution-matching regulariser, with a closed-form affine ridge probe for evaluation.
- Positive pairs: stationary Langevin diffusion on the latent manifold. It is an Ornstein–Uhlenbeck process for the Gaussian and Brownian motion for uniform compact worlds.
- Regularisers: full-dimensional heat-kernel MMD, with a Mehler kernel for the Gaussian, a sphere heat kernel, and a product torus kernel. These are matched to each target.
- Baselines: SIGReg for the Gaussian in the high-dimensional experiment, and a Matérn-3/2 MMD Gaussian control.
- KSD is only cited (KerJEPA, Liu 2016) and is not used.
- Data: 128×128 procedurally rendered robot images (2-D worlds). The high-dimensional experiment uses delayed trajectories of a two-particle Morse system.
- Latent dimensions: d = 4 to 128, with a d+2 ambient embedding.
- Encoder: EdgeNeXt-XXS, ImageNet-pretrained.

**Overlap with this study**
- Direct overlap with H3 in spirit: a prior matched to a non-Gaussian latent law beats a Gaussian target on linear identifiability.
- It uses MMD, not KSD, and targets are uniform on sphere or torus, not uniform or Laplace content laws.
- Temporal pairs are treated as latent diffusion. This is closest to our temporal-JEPA arm.
- A d versus d+2 rank-mismatch comparison is a form of dimensional bottleneck.
- No nuisance variables, no injected noise (information bottleneck), no 3DShapes, CIFAR or Moving-MNIST, and no leakage metric.

**Open gap**
- KSD or sliced-KSD with IMQ kernels.
- Nuisance independent across views, and leakage measured.
- Joint sweep of embedding dimension d and noise σ.
- Natural or standard benchmarks with known factors (3DShapes).
- Euclidean non-Gaussian (uniform or Laplace) content laws, which fall outside the Klindt Gaussian-uniqueness setting.
- VICReg, EMA and stop-gradient baselines.

**BibTeX**
```bibtex
@misc{nicollier2026gaussianworldslatentgeometry,
      title={Beyond Gaussian Worlds: Latent Geometry Matters for JEPAs}, 
      author={Léo Nicollier and Enric Meinhardt-Llopis and Marc Pic and Pablo Musé and Gabriele Facciolo},
      year={2026},
      eprint={2609.21656},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.21656}, 
}
```

**Table row**
- regulariser: heat-kernel MMD (matched); SIGReg and Matérn-MMD as Gaussian baselines
- sliced/full: full (SIGReg is sliced)
- prior: Gaussian, uniform on sphere, uniform on torus and Clifford torus (matched or mismatched)
- nuisance: — (not treated)
- dimensional bottleneck: partial (d versus d+2 ambient rank, a Gaussian rank bottleneck)
- information bottleneck: — (no injected noise)
- identifiability: linear, up to orthogonal or linear map (population theory plus affine-probe R²)
- objective: JEPA with diffusion positive pairs (temporal-like in the trajectory experiment)
- data: rendered images (procedural robot) and simulated trajectories
