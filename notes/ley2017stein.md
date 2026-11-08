# ley2017stein

Verified: title, authors, year, venue, volume, pages via Crossref (DOI 10.1214/16-ps278); title, authors, abstract via arXiv:1408.2998 page. The arXiv BibTeX API was rate-limited, so the BibTeX below is Crossref's. Only the abstract was read, not the body sections.

## Claim
The survey builds one general Stein-operator framework for univariate distributions, covering continuous (differential) and discrete (difference) cases. Operators are treated as acting on pairs of functions, which gives a shared toolkit for comparing two distributions. The authors derive general approximation bounds and apply them to several target pairs.

## Method
Nothing is regularised or measured on latents. It is theory: a density-based Stein operator for a 1-D target, with the score as the key ingredient, plus Stein-equation solution bounds. Examples in the abstract: normal vs normal, Rademacher sums vs normal, normal vs Student t, and maxima vs exponential, Fréchet and Gumbel. No kernel, and no sliced or full variant.

## Overlap with this study
- Univariate Stein operators for non-Gaussian targets (Laplace, smoothed uniform) can be built from the general density/score recipe. This is my inference; I read only the abstract.
- Sliced KSD and SIGReg act on 1-D projections, so a univariate operator is the natural building block.
- No JEPA, autoencoder, nuisance, bottleneck, identifiability or real-data content.

## Open gap
It has no kernelised discrepancy (KSD/IMQ), no multivariate or sliced extension, and no learning setting. The study can check empirically whether score operators for Laplace and smoothed uniform priors work as sliced-KSD regularisers on JEPA latents with known factors.

## BibTeX
```bibtex
@article{Ley_2017, title={Stein’s method for comparison of univariate distributions}, volume={14}, ISSN={1549-5787}, url={http://dx.doi.org/10.1214/16-ps278}, DOI={10.1214/16-ps278}, number={none}, journal={Probability Surveys}, publisher={Institute of Mathematical Statistics}, author={Ley, Christophe and Reinert, Gesine and Swan, Yvik}, year={2017}, month=Jan, pages={1–52} }
```

**Table row**
- regulariser: — (Stein operators, theory only)
- sliced/full: — (univariate)
- prior: general univariate targets (normal, Student t, extreme-value laws in examples)
- nuisance: — (not treated)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: other (probability theory)
- data: — (none)
