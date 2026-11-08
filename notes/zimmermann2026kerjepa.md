# KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning (Zimmermann, Wiltzer, Szeto, Alvarez-Melis, Mackey; arXiv:2512.19605, preprint, v2)

Verification: title, authors and year confirmed from the arXiv BibTeX record; the arXiv entry lists no venue. Content taken from the arXiv HTML text. The last ~30k characters (appendix proofs) were not read, so anything only in the appendix is unchecked.

## Claim
SIGReg (from LeJEPA) can be read as a sliced MMD to an isotropic Gaussian prior with a particular heavy-tailed kernel. The paper generalises this to a family of kernel-discrepancy regularisers (MMD and KSD) with other kernels and priors. It also derives closed-form (analytic) limits over all slices, which are more stable in high dimension than Monte Carlo slicing.

## Method
- Discrepancies: MMD and KSD. A slice-free estimator of a sliced KSD is given (Theorem 5).
- Variants: unsliced (ambient space), finite-sliced (e.g. 1024 random slices) and analytic (all slices).
- Kernels: Gaussian and IMQ are the main ones. IMQ approximates the SIGReg-equivalent kernel in high dimension (stated for d >= 20).
- Priors: Appendix B covers Gaussian, Laplace and Student-t. Experiments use Gaussian and Laplace, with negligible downstream difference in unsliced runs.
- Data: ImageNette (ViT-S/8), ImageNet-1k (ViT-S/B/L, EMA student-teacher, compared with DINO and iBOT), ADE20K linear segmentation, and DAVIS-2017 video segmentation with frozen image features.

## Overlap with this study
- Direct precedent for KSD and MMD with IMQ or Gaussian kernels, sliced versus full, as JEPA regularisers.
- Non-Gaussian priors (Laplace) are used, but only as a prior swap. KSD toward matched non-Gaussian priors as a diagnostic is not studied.
- Nuisance is not modelled. Augmentations are said to decide which features are learned.
- There is no embedding-dimension versus injected-noise bottleneck analysis. Only the dimension of the sliced or ambient space matters, for slicing quality.
- Identifiability is not analysed beyond the characteristic-kernel property.
- Temporal JEPA and autoencoders are not studied. DAVIS is evaluation only.
- Data are natural images only, with no known generative factors.

## Open gap
No controlled test of paired views with shared content and independent nuisance. No measurement of nuisance leakage or linear identifiability. No d versus sigma bottleneck sweep. No comparison to VICReg or SIGReg under noise. No rendered or known-factor data (3DShapes, Moving-MNIST). No temporal JEPA, and no KSD toward non-Gaussian priors that match the data.

## BibTeX
```bibtex
@misc{zimmermann2026kerjepakerneldiscrepancieseuclidean,
      title={KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning}, 
      author={Eric Zimmermann and Harley Wiltzer and Justin Szeto and David Alvarez-Melis and Lester Mackey},
      year={2026},
      eprint={2512.19605},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.19605}, 
}
```

**Table row**
- regulariser: MMD and KSD (Gaussian, IMQ kernels); SIGReg recovered as sliced MMD
- sliced/full: both (unsliced, finite-sliced, analytic all-slices)
- prior: Gaussian, Laplace (Student-t in Appendix B, not run)
- nuisance: not treated (augmentations implicit)
- dimensional bottleneck: — (only slicing dimension effects)
- information bottleneck: —
- identifiability: — (characteristic kernel only)
- objective: JEPA (joint-embedding, EMA student-teacher)
- data: natural (ImageNette, ImageNet-1k, ADE20K, DAVIS-2017)
