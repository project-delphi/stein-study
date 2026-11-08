# Gong, Li & Hernández-Lobato (2021), Sliced Kernelized Stein Discrepancy

Status: VERIFIED (title, authors, year confirmed on arXiv 2006.16531 and the mlanthology ICLR 2021 record; ICLR 2021 poster page iclr.cc/virtual/2021/poster/2803 appears in search results; no page range seen).

**Claim**
Kernelized Stein discrepancy with an RBF kernel degrades in high dimension. The paper projects both the score and the test input onto 1-D directions, so kernels only see scalars. It proves the result is a valid discrepancy and gets a goodness-of-fit (GOF) test, a training loss, and a sliced SVGD.

**Method**
Score directions r come from an orthogonal basis (the standard basis in practice). Each r gets its own test direction g_r, which is optimised by Adam from random initialisation, not random. max SKSD-g optimises g only. max SKSD-rg also optimises r. Full SSD with random uniform directions on the sphere is defined but called impractical. Kernels are RBF with the median heuristic. Deep kernels are only sketched. Estimation uses a U-statistic and bootstrap thresholds. The IMQ kernel appears only in the RFSD baseline.

**Uses**
- GOF: Gaussian, multivariate-t and diffusion benchmarks, plus a 50-d RBM.
- Training: ICA by directly minimising max SKSD.
- S-SVGD: amortised, in a VAE on binarised MNIST. It also covers a BNN and SGHMC step-size selection.
- The paper does not use KSD as a latent regulariser toward a Gaussian prior. The VAE use is particle inference on the posterior.

**Overlap with this study**
It gives the theory for a sliced KSD, with optimised rather than random slices. It also shows that full-dimensional KSD fails as d grows, which motivates comparing sliced and full KSD. It has no JEPA or temporal-JEPA, no nuisance, no dimensional or information bottleneck, no identifiability analysis, and no real data with known factors.

**Open gap**
The paper does not test slicing with IMQ or random directions (SIGReg-style) against optimised ones. It does not cover JEPA latents, nuisance leakage, bottleneck regimes (d vs σ), or linear identifiability on 3DShapes. It does not cover KSD toward non-Gaussian matched priors.

**BibTeX**
```bibtex
@inproceedings{gong2021sliced,
  title     = {Sliced Kernelized {Stein} Discrepancy},
  author    = {Gong, Wenbo and Li, Yingzhen and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://iclr.cc/virtual/2021/poster/2803},
  eprint    = {2006.16531},
  archivePrefix = {arXiv}
}
```

**Table row**
- regulariser: max SKSD (sliced KSD; also S-SVGD)
- sliced/full: sliced (orthogonal-basis score directions, optimised test directions)
- prior: —  (target p is the model or null, e.g. Gaussian, RBM, ICA; not a latent prior)
- nuisance: — (not treated)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: — (ICA fitting only, not analysed)
- objective: other (GOF test, ICA density fitting, amortised VAE posterior)
- data: synthetic (benchmarks, RBM, ICA) plus natural (binarised MNIST, UCI)
