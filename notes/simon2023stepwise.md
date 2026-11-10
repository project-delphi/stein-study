# simon2023stepwise

Verified: arXiv record (title, authors, 2023, comment "ICML '23") fetched via arXiv API. ICML venue rests on the arXiv comment only; proceedings page not separately fetched. Only the abstract was read, not the theorems, so the overlap judgements below are from the abstract and general knowledge of the paper.

## Claim
Joint-embedding SSL learns its embedding one dimension at a time, in discrete well-separated steps. A linearised Barlow Twins model, solved from small initialisation, learns the top eigenmodes of a contrastive kernel in sequence and has a closed-form final representation. The same stepwise behaviour shows up in ResNets trained with Barlow Twins, SimCLR and VICReg, so kernel PCA is proposed as the SSL analogue of kernel regression.

## Method
Training dynamics of a linearised, infinite-width Barlow Twins model with a cross-correlation (decorrelation) loss. The analysis is full-covariance and second-order; there is no kernel or Stein regulariser. Empirical checks use deep ResNets with Barlow Twins, SimCLR and VICReg.

## Overlap with this study
Only loosely related: a linear, closed-form theory of a variance/covariance-style SSL regulariser. It is about dynamics and the eigenmode spectrum, not about a fixed-point threshold.
- T1 (nuisance enters only above lambda* proportional to alpha, only when d exceeds the content dimension): not in the abstract. The paper's stepwise rank growth, where more dimensions are learned as training proceeds, is conceptually adjacent to a dimension-vs-content effect. There is no nuisance model, no lambda trade-off, and no threshold result that I read.
- T2 (kernel regularisers KSD/MMD/SIGReg, VICReg, learned linear predictor optimum): VICReg is only an empirical check. No KSD, MMD or SIGReg, and no predictor.
- T3 (predictor-input noise): not covered.
- T4 (rotation invariance vs ICA identifiability): not covered. Barlow Twins and kernel-PCA solutions are defined only up to rotation, which is the same background fact.
- No Stein, JEPA, temporal, autoencoder or real-data-with-known-factors content.

## Open gap
No nuisance or augmentation-resampling model, no phase-transition threshold in regulariser strength, no kernel or distribution-matching regularisers, no predictor, and no identifiability analysis. The study could add all of these, and could ask whether the stepwise ordering interacts with the lambda* threshold.

## BibTeX
```
@misc{simon2023stepwise,
  title={On the Stepwise Nature of Self-Supervised Learning},
  author={James B. Simon and Maksis Knutins and Liu Ziyin and Daniel Geisz and Abraham J. Fetterman and Joshua Albrecht},
  year={2023},
  eprint={2303.15438},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2303.15438},
  note={ICML 2023}
}
```
