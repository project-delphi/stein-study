# Wu 2026 VISReg (VERIFIED: title, authors, year via arXiv BibTeX and abs page; no venue listed, preprint)

**Claim**
VICReg's covariance term only controls second-order statistics, and SIGReg's single Gaussian-matching loss is inflexible and has vanishing gradients near collapse. VISReg replaces covariance with a sliced-Wasserstein sketching loss for full distributional shape and keeps a separate variance term for scale. It reports linear scaling, robustness on low-quality, long-tailed and low-rank data, and strong OOD results after ImageNet-1K/22K pretraining.

**Method**
Variance term (scale) + invariance term (paired views) + sliced-Wasserstein sketching term (shape), decoupling scale from shape. Sliced, not full. Target law: from the abstract, the sketching family aligns to an isotropic Gaussian; VISReg's exact target was not read in the body. Only the abstract was read.

**Overlap with this study**
Direct baseline family: SIGReg/VICReg-style distributional regularisation of JEPA latents. Uses sliced Wasserstein, not Stein/KSD/MMD. The abstract does not mention nuisance, bottleneck, identifiability or temporal data.

**Open gap**
No kernel/Stein discrepancy, non-Gaussian priors, nuisance leakage, d-vs-sigma bottleneck, identifiability test, or known-factor data. The study can cover all of these.

**BibTeX**
```bibtex
@misc{wu2026visregvarianceinvariancesketchingregularizationjepa,
      title={VISReg: Variance-Invariance-Sketching Regularization for JEPA training},
      author={Haiyu Wu and Randall Balestriero and Morgan Levine},
      year={2026},
      eprint={2606.02572},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.02572},
}
```

**Table row**
- regulariser: variance + sliced-Wasserstein sketching (+ invariance)
- sliced/full: sliced
- prior: isotropic Gaussian (per abstract's framing; body unchecked)
- nuisance: — (not treated in abstract)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: JEPA
- data: natural (ImageNet-1K/22K, OOD benchmarks)

**Full-text keyword check (added during P0 synthesis, arXiv HTML v1):** the body confirms the target is an isotropic Gaussian prior, matched by the Sliced Wasserstein distance (closed-form 1-D W2) plus a variance term. No occurrence of "nuisance" or "identifiab"; "bottleneck" appears once, about projection dimension limiting in-domain accuracy (an empirical ablation, not a bottleneck study). "Stein" appears only inside the word "Wasserstein". Video appears only in PCA visualisations against DINOv1.
