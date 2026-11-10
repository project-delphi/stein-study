# Comon 1994, Independent component analysis, a new concept?

Status: VERIFIED. Crossref record (DOI 10.1016/0165-1684(94)90029-9) confirms title, sole author, Signal Processing 36(3), April 1994, pp. 287-314. Theorem numbers below were read in the HAL full text (hal-00417283).

## Claim
Defines ICA as finding a linear map of a random vector whose outputs are as statistically independent as possible, framed through contrast functions (Kullback-type and cumulant-based). Shows the solution is unique up to scaling and permutation when at most one source is Gaussian, and that pairwise independence of outputs already suffices.

## Method
Theory plus algorithms, no learned latents. After whitening, the remaining freedom is an orthogonal matrix. Contrast functionals over output densities are maximised, with higher-order (fourth-order) cumulant contrasts and a pairwise (Jacobi-rotation style) algorithm. Identifiability uses the Darmois/Marcinkiewicz lemmas (Appendix A.7).

## Overlap with this study
Supports T4 as follows. Theorem 11: x has independent components, at most one Gaussian, none degenerate (densities not a point mass); C orthogonal, z = Cx. Then pairwise independence of z, mutual independence of z, and C = AP (A diagonal, P permutation) are equivalent. Corollary 13 (identifiability, noiseless) gives F = MAP for discriminating contrasts. The paper also states (eq. 1.2 area) that for unit-covariance x and z, C may be any unitary matrix without the non-Gaussianity condition, which is the rotation-invariance half of T4. Note the assumptions: it is stated for an orthogonal C after whitening, with independent sources; the general-M version is reached via whitening, and the Darmois-Skitovich name appears as "a variant of Darmois's theorem" (Theorem 10, Lemma 20). The paper does not discuss Gaussian-target regularisers, MCC, or orthogonal-fit R2; that link is the study's own. No JEPA, Stein/KSD/MMD, temporal or autoencoder content.

## Open gap
No finite-sample or approximate-independence version (a regulariser only reaches small divergence, not exact independence), no nuisance variables, no noise (noiseless only), no learned nonlinear encoders, and no link to a dimensional versus information bottleneck. The study can test empirically how far a matched product prior recovers coordinates (MCC) versus a Gaussian one (R2 only).

## BibTeX
```bibtex
@article{Comon_1994,
  title   = {Independent component analysis, A new concept?},
  volume  = {36},
  ISSN    = {0165-1684},
  url     = {http://dx.doi.org/10.1016/0165-1684(94)90029-9},
  DOI     = {10.1016/0165-1684(94)90029-9},
  number  = {3},
  journal = {Signal Processing},
  publisher = {Elsevier BV},
  author  = {Comon, Pierre},
  year    = {1994},
  month   = apr,
  pages   = {287--314}
}
```
