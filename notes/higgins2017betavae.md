# higgins2017betavae

Verification: VERIFIED for title, authors, year and venue (ICLR 2017) from the ML Anthology record, which links the OpenReview PDF (id Sy2fzU9gl). OpenReview was not fetched directly. Method details beyond the abstract are from memory of the paper, not a fresh PDF read.

**Claim**
Scaling the VAE's KL term by beta > 1 pushes the latent posterior toward the factorised prior N(0,I). This yields more disentangled, interpretable factors from raw images with no supervision. Beta trades reconstruction against a capacity and independence constraint.

**Method**
The objective is reconstruction plus beta times KL(q(z|x) || N(0,I)), a closed-form full KL (not sliced) with a diagonal Gaussian posterior. Disentanglement metric: fix one generative factor, encode pairs of images, and average the absolute latent differences. A linear classifier then predicts which factor was fixed, and its accuracy is the score. Datasets: a synthetic 2D-shapes set (the dSprites precursor), plus celebA, 3D faces and 3D chairs. Baselines are InfoGAN and DC-IGN.

**Overlap with this study**
This is the beta-VAE arm: a KL-to-prior regulariser on autoencoder latents. Beta acts as an information bottleneck. Nuisance is not modelled. It uses synthetic data with known factors, and the metric is a proxy for identifiability.

**Open gap**
It has no kernel or Stein discrepancy, no JEPA or temporal setting, and no nuisance-leakage measure. It gives no identifiability guarantee, and the metric depends on the classifier. 3DShapes is not evaluated.

**BibTeX**
```bibtex
@inproceedings{higgins2017iclr-beta,
  title     = {{Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework}},
  author    = {Higgins, Irina and Matthey, Lo{\"i}c and Pal, Arka and Burgess, Christopher P. and Glorot, Xavier and Botvinick, Matthew M. and Mohamed, Shakir and Lerchner, Alexander},
  booktitle = {International Conference on Learning Representations},
  year      = {2017},
  url       = {https://openreview.net/pdf?id=Sy2fzU9gl}
}
```
(Fields from the ML Anthology record; the url is the OpenReview PDF link given there. No arXiv or DBLP record fetched.)

**Table row**
- regulariser: beta-weighted KL to prior
- sliced/full: full (closed-form KL)
- prior: N(0,I)
- nuisance: not treated
- dimensional bottleneck: — (latent size fixed, not the lever)
- information bottleneck: yes (beta > 1 limits channel capacity)
- identifiability: empirical only (disentanglement metric, no theory)
- objective: autoencoder (VAE)
- data: synthetic (2D shapes) and rendered (3D faces, chairs), plus natural (celebA)
