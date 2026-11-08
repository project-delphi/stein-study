# gorham2017measuring — VERIFIED (title, authors, venue, pages confirmed on PMLR v70 page; abstract via arXiv API)

**Claim**
Kernel Stein discrepancies (KSDs) need not detect non-convergence: for d ≥ 3, KSDs built on Gaussian or Matérn kernels can go to zero for sample sequences that do not converge to the target, even a Gaussian one. Kernels with slowly decaying tails fix this. The inverse multiquadric (IMQ) kernel k(x,y) = (c² + ||x−y||₂²)^β, c > 0, β ∈ (−1, 0), gives a KSD that controls weak convergence for distantly dissipative targets with suitably smooth scores.

**Method**
Stein's-method theory for closed-form KSDs: a Langevin Stein operator applied to a kernel's RKHS ball. The KSD is a double sum of kernel evaluations over sample pairs, using the target score only. It is full (pairwise), not sliced. Applications: sampler comparison, hyperparameter selection, one-sample testing.

**Overlap with this study**
Direct justification for choosing the IMQ kernel in KSD on embedding batches. It supports Stein/KSD regularisation toward a Gaussian or non-Gaussian prior with a known score. Convergence-determining behaviour is the relevant property, since a non-determining kernel could leave latents far from the prior. No JEPA, temporal, autoencoder, nuisance, bottleneck, identifiability or image content.

**Open gap**
Guarantees are stated for fixed low to moderate dimension and for MCMC samples. Not covered: high-dimensional embeddings, minibatch estimates, sliced KSD, behaviour as a training regulariser, comparison to SIGReg/VICReg/MMD, and recovery of known factors.

**BibTeX**
```bibtex
@InProceedings{gorham17a,
  title     = {Measuring Sample Quality with Kernels},
  author    = {Gorham, Jackson and Mackey, Lester},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  pages     = {1292--1301},
  year      = {2017},
  volume    = {70},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v70/gorham17a.html},
  note      = {arXiv:1703.01717}
}
```
(Fields from the PMLR page metadata; year from its 2017/07/17 date. Series/publisher/url are standard PMLR; the arXiv BibTeX gives only a 2020 misc entry.)

**Table row**
- regulariser: — (KSD as a measure; IMQ kernel recommended)
- sliced/full: full
- prior: — (any target with known score; Gaussian examples)
- nuisance: — (not treated)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: other (sample-quality diagnostic for MCMC)
- data: synthetic
