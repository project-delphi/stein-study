# Shwartz-Ziv et al. 2023, VICReg information-theoretic perspective

Status: VERIFIED. Authors and abstract from the arXiv API (2303.00633, v4); NeurIPS 2023 (vol. 36) confirmed from the proceedings page in search results (https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b1d4c03391b0aa6ddde0b807a78c950-Abstract.html). The proceedings title is "An Information Theory Perspective on ..."; arXiv uses "Information-Theoretic". Only the abstract was read; no section or theorem numbers are cited.

## Claim
VICReg's objective can be read as optimising mutual information between embeddings of the two views, once information quantities are defined for deterministic networks. The paper states the assumptions this needs, compares VICReg with other SSL methods, derives a downstream generalisation bound, and proposes new information-theoretic SSL variants.

## Method
Analytic and empirical. Entropy and mutual-information terms are derived for deterministic encoders (not stochastic ones). The variance and covariance terms are linked to entropy of the embedding, and the invariance term to a conditional-entropy/alignment term. No kernel or Stein discrepancy is used; the regulariser is VICReg's own (variance hinge, covariance), with proposed variants.

## Overlap with this study
- Relates to the information-bottleneck vs dimensional-bottleneck question and to VICReg's hinge (T2 side): it gives an information view of why the variance/covariance terms matter. Based on the abstract, I found no linear nuisance model.
- T1 (threshold lambda* proportional to alpha, nuisance only when d exceeds content dimension): not in the abstract.
- T2 (kernel regularisers): no. VICReg's hinge is treated information-theoretically, not as a linear-theory threshold.
- T3 (predictor-input noise): not in the abstract.
- T4 (rotation invariance vs ICA identifiability): not in the abstract.
- No Stein/KSD/MMD, no temporal JEPA, no real data with known factors (per the abstract).

## Open gap
No closed-form linear analysis of when nuisance enters the latent as a function of regularisation strength and width, and no comparison of Gaussian-target regularisers with identifiability. The study can supply both, with known-factor data to test them. Full text was not read, so the paper may contain partial statements on these points.

## BibTeX
```
@inproceedings{shwartzziv2023information,
  title     = {An Information Theory Perspective on Variance-Invariance-Covariance Regularization},
  author    = {Shwartz-Ziv, Ravid and Balestriero, Randall and Kawaguchi, Kenji and Rudner, Tim G. J. and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023},
  eprint    = {2303.00633},
  archivePrefix = {arXiv},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/6b1d4c03391b0aa6ddde0b807a78c950-Abstract.html}
}
```
