# tian2021understanding

Verified: title, authors, year confirmed via arXiv API (2102.06810) and the PMLR page
(ICML 2021, PMLR 139, pp. 10268-10278). Read abstract and main-text theorem statements only.

## Claim
Non-contrastive SSL (BYOL, SimSiam) avoids collapse because of the interaction of a
learned linear predictor, stop-gradient, EMA and weight decay, not because of negatives.
In a linear/two-layer model the predictor's eigenspace aligns with the eigenspace of the
predictor-input correlation matrix F, and eigenmodes grow or die depending on weight decay
and relative predictor learning rate. This motivates DirectPred, which sets the linear
predictor from the eigendecomposition of F with no gradient training.

## Method
Gradient-flow analysis of a two-layer linear online network W, linear predictor Wp, and
EMA target Wa, with weight decay eta. Key results: Lemma 1 (dynamics), Theorem 1 (weight
decay balances predictor and online weights), Theorem 2 (stop-gradient is essential),
Theorem 3 (eigenspace alignment between Wp and F), plus an exact integral of motion
(invariant parabola s_j ~ p_j^2) and a per-mode ODE. Nothing is regularised with a
covariance, kernel or Stein penalty; collapse is controlled by dynamics. Experiments on
STL-10 and ImageNet.

## Overlap with this study
- T1 (threshold lambda* proportional to alpha for nuisance entering the embedding, only when
  d exceeds content dimension): not in the paper. No content/nuisance generative model, no
  Cov(z) - I penalty, no resampling probability alpha. Closest relative: the per-mode
  condition that p_j(...) - eta must be positive for a mode to grow (Section 3, the
  quantity governing collapse vs non-collapse), a weight-decay threshold on eigenmodes.
  Structurally similar (a threshold below which directions die) but a different mechanism.
- T2 (KSD/MMD/SIGReg/VICReg regularisers; learned linear predictor optimum): partly. The
  predictor side is related: Theorem 3 (eigenspace alignment) and DirectPred (Section 4)
  say the learned predictor ends up a function of F's eigenbasis, which bears on the
  "learned linear predictor's optimum". No kernel or variance/covariance regulariser is
  analysed.
- T3 (predictor-input noise): not treated as noise. Closest: the identity-plus-F structure
  and weight decay / EMA effects on predictor eigenvalues; no analysis of noise in the
  predictor input.
- T4 (rotation invariance vs ICA identifiability): not covered.
- No Stein/KSD/MMD, temporal JEPA, autoencoder latents, dimensional vs information
  bottleneck, or real data with known factors.

## Open gap
No account of what nuisance a regulariser lets into the embedding, no content/nuisance
model, no kernel or Gaussian-target regularisers, no identifiability analysis, and no
dependence on embedding dimension relative to content dimension. All are open for the study.

## BibTeX
```
@inproceedings{tian2021understanding,
  title     = {Understanding self-supervised Learning Dynamics without Contrastive Pairs},
  author    = {Tian, Yuandong and Chen, Xinlei and Ganguli, Surya},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {10268--10278},
  year      = {2021},
  publisher = {PMLR},
  eprint    = {2102.06810},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v139/tian21a.html}
}
```
(arXiv record: arXiv:2102.06810, cs.LG. Booktitle, volume, pages, publisher from the PMLR page.)
