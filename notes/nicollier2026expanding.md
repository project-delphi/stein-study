# Nicollier 2026 — Expanding SPHERE-JEPA

Status: VERIFIED (title, authors, year from arXiv BibTeX and abs page; arXiv preprint, no venue listed; the export API rate-limited, so the abstract came from the abs page).

**Claim**
Sliced uniformity regularisers on the hypersphere (SIGReg, SUSReg) inject gradient variance through random projections. Integrating the projections out gives a deterministic MMD. The authors build full-dimensional MMD, KSD and KDE-KL regularisers on the sphere that are more stable and converge faster. The choice of test also shapes the latent geometry.

**Method**
Embeddings are pushed toward the uniform law on the unit sphere. Objectives are MMD, KSD and KL (KDE-based). Kernels are rotation-invariant and built spectrally: Heat (smooth decay) and Bandlimited (sharp cutoff). The sliced versus full comparison is the central point. Evaluation is on ImageNet and Galaxy10, with procedural texture retrieval.

**Overlap with this study**
- Closest prior for full KSD/MMD versus sliced regularisation of JEPA-style SSL latents, and for the claim that slicing adds variance.
- Shares the KSD and MMD families, and the sliced/full contrast.
- Differs on the kernel (spectral sphere kernels, not IMQ) and on the target (uniform on the sphere, not Gaussian or matched non-Gaussian priors).
- I did not read the full text, so any treatment of nuisance, bottlenecks or identifiability is not confirmed. The abstract mentions none.

**Open gap**
- No controlled nuisance or leakage test, and no paired views with independent nuisance.
- No dimension-versus-noise bottleneck sweep.
- No linear identifiability on data with known factors (3DShapes).
- No non-Gaussian matched-prior KSD.
- No temporal data.

**BibTeX**
```
@misc{nicollier2026expandingspherejepafamilystatistical,
      title={Expanding SPHERE-JEPA: A Family of Statistical Regularizers for the Hypersphere}, 
      author={Léo Nicollier and Enric Meinhardt-Llopis and Max Dunitz and Marc Pic and Pablo Musé and Gabriele Facciolo},
      year={2026},
      eprint={2606.17603},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.17603}, 
}
```

**Table row**
- regulariser: full MMD, KSD, KDE-KL on the sphere (compared with sliced SIGReg/SUSReg)
- sliced/full: full (sliced as baseline)
- prior: uniform on the hypersphere
- nuisance: — (not stated in abstract)
- dimensional bottleneck: — (not stated)
- information bottleneck: —
- identifiability: —
- objective: JEPA-style SSL
- data: natural (ImageNet, Galaxy10), procedural textures

**Full-text keyword check (added during P0 synthesis, arXiv HTML v1):** the body contains no occurrence of "nuisance", "identifiab", "bottleneck", "IMQ", "video"/"temporal", or 3DShapes/dSprites. KSD is built on the hypersphere (Riemann–Stein framework, Appendix B) toward the uniform law; experiments are ImageNet-100, Galaxy10 and instance-level texture retrieval. "Laplace" appears only as the Laplace–Beltrami operator.
