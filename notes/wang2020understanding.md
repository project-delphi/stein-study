# wang2020understanding (VERIFIED: PMLR v119 page and arXiv:2005.10242)

**Claim**
Contrastive learning with normalised features asymptotically optimises two properties: alignment (positive pairs map close) and uniformity (features spread evenly on the sphere). Both can be measured and optimised directly. Together they track downstream accuracy better than either alone, so uniformity is not a side effect; collapse-avoidance matters.

**Method**
Alignment loss: expected distance (power alpha) between positive-pair embeddings. Uniformity loss: log of the mean Gaussian potential, exp(-t||x-y||^2), over pairs. It is minimised by the uniform law on the sphere, a non-Gaussian target. It is a pairwise, full (not sliced) kernel statistic, close to an MMD-type energy. Evaluated on image and text benchmarks.

**Overlap with this study**
Closest to kernel/MMD regularisation of latents toward a fixed prior. Uniformity is a distributional regulariser; alignment is the JEPA-like invariance term. No Stein/KSD, no temporal JEPA, no nuisance treatment, no bottleneck analysis, no identifiability claim.

**Open gap**
Compare a spherical-uniform target with Gaussian targets (SIGReg, IMQ-KSD) under JEPA bottlenecks. Measure nuisance leakage and identifiability on real data with known factors.

**BibTeX**
```bibtex
@inproceedings{wang2020understanding,
  title     = {Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author    = {Wang, Tongzhou and Isola, Phillip},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {9929--9939},
  year      = {2020},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v119/wang20k.html}
}
```

**Table row**
- regulariser: Gaussian-potential uniformity + alignment loss
- sliced/full: full (pairwise)
- prior: uniform on hypersphere
- nuisance: — (not treated explicitly; invariance to augmentations only)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: — (not claimed)
- objective: other (contrastive)
- data: natural (CIFAR, STL-10, ImageNet subsets, BookCorpus)
