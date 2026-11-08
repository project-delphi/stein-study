# Bardes, Ponce, LeCun 2022 - VICReg

Verification: title, authors, year confirmed via arXiv API/BibTeX (2105.04906). Venue ICLR 2022 confirmed by the arXiv comment "Accepted at ICLR 2022" and the HAL record (hal-03541297, ICLR 2022, Apr 2022, online). The OpenReview page was blocked by a bot challenge, so no OpenReview id or URL is given.

**Claim**
VICReg prevents collapse in joint-embedding self-supervised learning with explicit loss terms, not architectural tricks. It combines an invariance term between paired views with a per-dimension variance hinge and a covariance decorrelation penalty. It matches the state of the art on downstream tasks, and the variance term also stabilises other methods.

**Method**
It uses three terms on batch embeddings of two views. Invariance is the MSE between paired embeddings. Variance is a hinge keeping each dimension's batch standard deviation above a threshold. Covariance pushes off-diagonal entries of the embedding covariance toward zero. It uses only first and second moments: no kernel, no prior, no slicing. The loss is full (all-pairs covariance) and is applied to the encoder outputs and the expander (projector) outputs.

**Overlap with this study**
- VICReg is a direct baseline regulariser for the JEPA comparison.
- It matches moments only, so the contrast with KSD/MMD/SIGReg (full distribution match to a Gaussian prior) is clear.
- Nuisance is not modelled. Views differ only through augmentations, and invariance is the only mechanism that discards nuisance.
- The expander dimension acts as a dimensional bottleneck. There is no injected noise, so no information bottleneck.
- No identifiability analysis.
- Data are natural images (ImageNet), with no known factors.

**Open gap**
- Nuisance leakage and linear identifiability under controlled factors (3DShapes, Moving-MNIST).
- The d versus sigma bottleneck trade-off.
- Whether moment matching is enough compared with Stein or MMD distribution matching.
- Temporal views.

**BibTeX**
The ICLR 2022 version has no proceedings page I could fetch. The fields below come from the verified arXiv record and the confirmed venue.
```bibtex
@inproceedings{bardes2022vicreg,
  title     = {{VICReg}: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author    = {Adrien Bardes and Jean Ponce and Yann LeCun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  eprint    = {2105.04906},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2105.04906}
}
```

**Table row**
- regulariser: VICReg (variance hinge + covariance decorrelation + invariance MSE)
- sliced/full: full (no slicing)
- prior: — (moment constraints only, no explicit prior)
- nuisance: treated only implicitly, through view augmentations and the invariance term
- dimensional bottleneck: embedding/expander dimension only; no dedicated analysis
- information bottleneck: —
- identifiability: — (not analysed)
- objective: JEPA-style joint embedding (siamese, two views)
- data: natural (ImageNet, with transfer evaluations)
