# mijoule2023stein

Verified via arXiv 2101.05079 (v2) and Crossref (DOI 10.1214/22-ejp883). Note: the author list is Mijoule, Raič, Reinert, Swan (four authors), published in Electronic Journal of Probability 28, 2023.

**Claim**
The paper gives a general framework for Stein's density method on R^d. Any continuous density gets a canonical operator and Stein class, plus a family of "standardizations", which yield Stein identities and characterizations. The score-based and Stein-kernel versions are highlighted, and neither needs the normalizing constant.

**Method**
Theory only, with no learning and no regularisation. The construction starts from the density and builds the canonical operator and Stein class. Standardizations then give the score-function operator and Stein-kernel operators. A new Stein kernel definition is linked to mass transport, with explicit formulas for elliptical laws. Stein discrepancy is used to bound 2-Wasserstein and 1-Wasserstein distances. A weak Stein equation is bounded for Lipschitz test functions under a Poincaré constant. Examples include Gaussian, Student, normal-gamma, Azzalini-Dalla Valle, Bayesian posteriors and copulas. No RKHS kernel (IMQ) and no slicing appear in the abstract.

**Overlap with this study**
It supplies the theoretical base for score-based Stein operators that apply to any continuous density with an unnormalised target. This supports using KSD against non-Gaussian priors such as Laplace or a smooth uniform surrogate. The Stein-discrepancy-to-Wasserstein link is relevant to the study's interpretation. It does not cover JEPA, autoencoders, nuisance, bottlenecks, identifiability, or image data.

**Open gap**
It has no kernelised (KSD) estimator, no finite-sample or sliced high-dimensional behaviour, and no application to learned latents. The study can test empirically whether KSD toward non-Gaussian priors under JEPA bottlenecks recovers known factors. Hard-support priors such as uniform need a smooth surrogate, and this is not checked here.

**BibTeX**
```bibtex
@article{Mijoule_2023,
  title={Stein’s density method for multivariate continuous distributions},
  volume={28},
  ISSN={1083-6489},
  url={http://dx.doi.org/10.1214/22-ejp883},
  DOI={10.1214/22-ejp883},
  number={none},
  journal={Electronic Journal of Probability},
  publisher={Institute of Mathematical Statistics},
  author={Mijoule, G. and Raič, M. and Reinert, G. and Swan, Y.},
  year={2023},
  month=Jan,
  pages={1–40}
}
```

**Table row**
- regulariser: — (Stein operators / Stein discrepancy theory)
- sliced/full: —
- prior: any continuous density on R^d (examples: Gaussian, Student, elliptical, normal-gamma)
- nuisance (treated? how): —
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective (JEPA / temporal JEPA / autoencoder / other): other (probability theory)
- data (synthetic / rendered / natural): —
