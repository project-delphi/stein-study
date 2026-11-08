# pu2017vae

Status: VERIFIED. Title and authors confirmed on the NeurIPS proceedings page, which lists the first author as "Yuchen Pu". arXiv and the paper text spell it "Yunchen". Venue is NIPS/NeurIPS 2017 (vol. 30, pp. 4236-4245) and arXiv:1704.05155.

## Claim
Stein variational gradient descent (SVGD) replaces the usual parametric encoder density in VAE training. Posterior samples are moved by a kernel-based transport map, so the encoder needs no closed-form density. An importance-weighted version gives a multi-sample bound, and SVGD can also place a posterior over decoder parameters.

## Method
SVGD minimises KL(q||p) by pushing particles along an RKHS direction. Here z-particles (and optionally decoder-parameter particles θ) are moved towards p(z|x). A neural sampler f_η(x, ξ), fed isotropic Gaussian noise ξ, is trained to match the moved particles. It supplies amortisation but has no density. The kernel is an RBF with a median-heuristic bandwidth, with M=100 particles and k=50 for the importance weighting. It is the full kernel, not sliced. Stein is the transport tool, not a discrepancy to the prior.

## Overlap with this study
Thin. It is a Stein-based VAE in the KL family and uses no KSD or MMD penalty. It has no aggregate-posterior matching, no IMQ kernel, no JEPA or temporal objective, no nuisance handling and no identifiability analysis. It is useful as a contrast: the SVGD/KL posterior route versus a KSD-to-prior route.

## Open gap
It never penalises the aggregate posterior with a kernel Stein discrepancy. It has no non-Gaussian prior, no nuisance-leakage measure, no bottleneck analysis and no rendered data with known factors such as 3DShapes.

## BibTeX
```bibtex
@inproceedings{pu2017vae,
  title     = {VAE Learning via Stein Variational Gradient Descent},
  author    = {Pu, Yunchen and Gan, Zhe and Henao, Ricardo and Li, Chunyuan and Han, Shaobo and Carin, Lawrence},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  pages     = {4236--4245},
  year      = {2017},
  eprint    = {1704.05155},
  archivePrefix = {arXiv},
  url       = {https://papers.neurips.cc/paper/2017/hash/443dec3062d0286986e21dc0631734c9-Abstract.html}
}
```

**Table row**
- regulariser: SVGD transport of posterior particles (KL objective, importance-weighted); no explicit penalty
- sliced/full: full (RBF kernel, median bandwidth)
- prior: simple prior p(z); not varied. Gaussian q0 for initial particles and noise
- nuisance: — (not treated)
- dimensional bottleneck: — (50-d latent, not studied)
- information bottleneck: — (KL term only, implicit)
- identifiability: — (none)
- objective: autoencoder (VAE / IWAE variants)
- data: synthetic (Gaussian mixture toy) and natural (MNIST, text corpora, ImageNet semi-supervised)
