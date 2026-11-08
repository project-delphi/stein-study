# chwialkowski2016kernel — VERIFIED
(Title, authors, year confirmed via arXiv API/BibTeX; venue, volume, pages via PMLR v48 page metadata.)

**Claim**
A goodness-of-fit test compares samples to a target density known only up to normalisation. It uses a kernel Stein discrepancy (KSD): the largest mean of a Stein-operator-transformed RKHS function over the samples. Only the target's score (log-density gradient) is needed, and the statistic is computed in closed form in quadratic time.

**Method**
The Stein operator is applied to a vector-valued RKHS ball (product RKHS, Oates et al. style), so the target expectation is zero. The statistic is a V-statistic built from the score and the kernel. Its null quantiles come from a wild bootstrap: a Markov chain of random sign flips with flip probability a_n, which also handles autocorrelated samples (MCMC). Kernel: Gaussian in the experiments; IMQ is not used. Full (not sliced) KSD, evaluated in the original space. Experiments: Student-t chains, GP criticism, MCMC bias.

**Overlap with this study**
Defines the KSD the study applies to embedding batches. The score of a Gaussian prior is closed-form, which matches the Gaussian-prior case. Gives the wild bootstrap for correlated samples, which is relevant to temporal JEPA. Nothing on JEPA, autoencoders, nuisance, bottlenecks, identifiability or image data.

**Open gap**
No use as a training regulariser (gradient through the statistic, minibatch bias), no sliced variant, no IMQ or kernel-choice analysis in high dimension, no non-Gaussian matched priors for learned latents, no link to factor recovery on data with known factors.

**BibTeX**
```bibtex
@inproceedings{chwialkowski2016kernel,
  title     = {A Kernel Test of Goodness of Fit},
  author    = {Chwialkowski, Kacper and Strathmann, Heiko and Gretton, Arthur},
  booktitle = {Proceedings of The 33rd International Conference on Machine Learning},
  pages     = {2606--2615},
  year      = {2016},
  volume    = {48},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v48/chwialkowski16.html},
  eprint    = {1602.02964},
  archivePrefix = {arXiv}
}
```

**Table row**
- regulariser: full KSD (test statistic, not a regulariser)
- sliced/full: full
- prior: arbitrary target with known score (unnormalised)
- nuisance (treated? how): — (no)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: other (goodness-of-fit testing)
- data: synthetic
