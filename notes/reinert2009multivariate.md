# Reinert & Röllin 2009 — Multivariate normal approximation, exchangeable pairs (VERIFIED)

Verified via Crossref (DOI 10.1214/09-aop467, AoP 37(6), 2150–2173) and arXiv:0711.1082 (v3).

**Claim**
The paper builds a multivariate exchangeable-pairs version of Stein's method. It bounds distances to possibly singular (rank-deficient) multivariate normals under a general linearity condition. It also gives an embedding that lifts a statistic into a higher-dimensional space, so the approach applies when the statistic does not fit directly. Examples are runs on the line and double-indexed permutation statistics.

**Method**
Pairs (W, W') are exchangeable, and a linearity condition links E[W'−W | W] to W. The condition is general, so the covariance need not be invertible. The Stein characterisation gives distributional distances to the target. There is no kernel, prior or learned model, and no sliced version. This is probability theory and nothing is trained.

**Overlap with this study**
- Singular Gaussian targets are the closest match to the "oracle" arm, KSD toward a rank-k Gaussian in a larger ambient space. This is theoretical support for a Stein-type distance to a degenerate normal.
- Paired views resemble exchangeable pairs, but the paper's pairs are constructed from a statistic and not from content with independent nuisance.
- It says nothing about JEPA, autoencoders, MMD, KSD or image data.
- Nuisance, bottlenecks and identifiability are not covered.

**Open gap**
Whether exchangeable-pair Stein arguments can bound the KSD or MMD of JEPA latents, with view pairs as the exchangeable pair. Also open are the effect of nuisance on the linearity condition, and finite-sample behaviour on real data with known factors.

**BibTeX**
```bibtex
@article{Reinert_2009, title={Multivariate normal approximation with Stein’s method of exchangeable pairs under a general linearity condition}, volume={37}, ISSN={2168-894X}, url={http://dx.doi.org/10.1214/09-aop467}, DOI={10.1214/09-aop467}, number={6}, journal={The Annals of Probability}, publisher={Institute of Mathematical Statistics}, author={Reinert, Gesine and Röllin, Adrian}, year={2009}, month=Nov, pages={2150–2173} }
```

**Table row**
- regulariser: — (Stein exchangeable-pairs bounds, theory only)
- sliced/full: —
- prior: possibly singular multivariate normal
- nuisance: — (not treated)
- dimensional bottleneck: — (singular covariance is rank-deficiency, but not a bottleneck)
- information bottleneck: —
- identifiability: —
- objective: other (probability theory)
- data: — (none; theoretical examples only)
