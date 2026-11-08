# kuang2026rectified

VERIFIED (title, authors, year, eprint via arXiv BibTeX; abstract and method read from arXiv HTML. Venue: arXiv preprint only; no acceptance confirmed. The arXiv Atom API was rate-limited, so the record is the BibTeX plus the HTML page. Appendices, including the projector-dimension ablation J.2, were not read.)

## Claim
Rectified LpJEPA, by Kuang, Dagade, Rudner, Balestriero and LeCun, replaces LeJEPA's isotropic-Gaussian target with a Rectified Generalized Gaussian (RGG). Its loss, RDMReg, pushes JEPA embeddings toward that target. Rectification adds a point mass at zero, which controls expected l0 sparsity, and p = 2 recovers a rectified Gaussian. The result is sparse, non-negative representations with good sparsity-accuracy tradeoffs.

## Method
RDMReg is a sliced two-sample loss. Batch features and samples from the product-RGG target (parameters mu, sigma, p) are projected on random unit directions, sorted, and compared by squared distance, which is a sliced-Wasserstein-type loss. Feature-covariance eigenvectors can replace random directions. It is added to a view-invariance term. The target family is fixed by hyperparameters (sparsity and p), not fitted to data. The theory covers Cramér–Wold matching, maximum entropy under l_p and support constraints, and a conditional link to non-negative VCReg.

## Overlap with this study
Same family: a distributional prior on JEPA latents, a non-Gaussian target, and sliced matching. It uses a sorting/Wasserstein discrepancy, not KSD, MMD or Stein. No Stein/KSD. No nuisance, identifiability or injected-noise analysis seen. Data are natural images only.

## Open gap
No paired views with independent nuisance, no leakage or linear-identifiability metrics, no d-versus-sigma bottleneck sweep, no kernel (IMQ) comparison. Targets are not matched to known content laws, and no data with known factors are used. Our study can test these.

## BibTeX
```bibtex
@misc{kuang2026rectifiedlpjepajointembeddingpredictive,
      title={Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations},
      author={Yilun Kuang and Yash Dagade and Tim G. J. Rudner and Randall Balestriero and Yann LeCun},
      year={2026},
      eprint={2602.01456},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.01456},
}
```

**Table row**
- regulariser: RDMReg (sorted-projection two-sample loss, sliced-Wasserstein-type)
- sliced/full: sliced (random or eigenvector directions)
- prior: product of i.i.d. Rectified Generalized Gaussian (mu, sigma, p); fixed, not data-matched
- nuisance: — (not treated)
- dimensional bottleneck: — (projector-dim ablation in App. J.2, unread)
- information bottleneck: —
- identifiability: — (only Cramér–Wold matching and max-entropy theory)
- objective: JEPA
- data: natural (ImageNet-100, CIFAR-100, transfer/OOD; ImageNet-1K in appendix)
