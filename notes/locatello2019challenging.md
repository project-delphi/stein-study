# locatello2019challenging (VERIFIED: arXiv record and PMLR v97 page)

**Claim**
Unsupervised disentanglement is impossible in general: for any factorised latent distribution there are infinitely many equally likely generative models whose latents are entangled, so the data distribution cannot pick out the factors. A study of over 12,000 models shows that the methods enforce what their losses encourage. It also shows that good models cannot be picked out without supervision.

**Method**
The proof builds a measure-preserving map of a factorised latent that entangles every coordinate. The experiments cover beta-VAE, AnnealedVAE, FactorVAE, beta-TCVAE and DIP-VAE-I/II, all with factorised Gaussian priors. They are scored with BetaVAE, FactorVAE, MIG, DCI, SAP and Modularity metrics on seven datasets, including Shapes3D, dSprites, Cars3D and SmallNORB. Seeds and hyperparameters are varied, and downstream sample complexity is measured.

**Overlap with this study**
- Shapes3D with known factors.
- Motivates identifiability only through pairing or weak supervision, which matches our shared-content views.
- The regularisers are KL or total-correlation penalties on autoencoder latents, not Stein, KSD or MMD, and there is no JEPA.
- Nuisance is not modelled.
- Neither bottleneck type is analysed separately.

**Open gap**
- Test whether a kernel or Stein regulariser on JEPA latents gives identifiability once pairing supplies the inductive bias.
- Measure linear identifiability of content and nuisance leakage, which the paper's metrics do not capture.
- Separate a dimensional bottleneck from an information bottleneck.

**BibTeX**
```bibtex
@InProceedings{locatello2019challenging,
  title     = {Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
  author    = {Locatello, Francesco and Bauer, Stefan and Lucic, Mario and Raetsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages     = {4114--4124},
  year      = {2019},
  editor    = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume    = {97},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  eprint    = {1811.12359},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v97/locatello19a.html}
}
```

**Table row**
- regulariser: KL / total-correlation / moment-matching penalties (beta-VAE family, DIP-VAE)
- sliced/full: —
- prior: factorised Gaussian
- nuisance (treated? how): not treated
- dimensional bottleneck: —
- information bottleneck: KL weight (beta) acts implicitly
- identifiability: proved impossible without inductive biases
- objective: autoencoder (VAE)
- data: synthetic / rendered
