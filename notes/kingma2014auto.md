# kingma2014auto: Auto-Encoding Variational Bayes (Kingma & Welling, ICLR 2014)

Verification: title, authors, abstract from the arXiv API (1312.6114, first submitted 20 Dec 2013). Venue (ICLR 2014, Banff) from the UvA repository record plus a search listing of the conference proceedings; the OpenReview page was blocked by a browser check.

## Claim
Reparameterising the variational lower bound (ELBO) gives a low-variance estimator trainable by ordinary SGD. An encoder (recognition model) is fitted jointly with a decoder, which yields the VAE.

## Method
Maximise the ELBO: reconstruction term minus a per-sample KL from q(z|x) to a N(0,I) prior (closed form for Gaussian q). Sampling is z = mu + sigma*eps. No kernel and no slicing.

## Overlap with this study
Baseline for the optional VAE tier: the beta-VAE arm is this KL with a weight. Contrast with the KSD-to-prior and WAE-MMD arms, which regularise the aggregate posterior. No JEPA, temporal, nuisance or identifiability content.

## Open gap
Per-sample KL is not an aggregate-posterior match and assumes a Gaussian prior. Nuisance leakage and identifiability on 3DShapes against JEPA arms are unmeasured.

## BibTeX
```bibtex
@inproceedings{kingma2014auto,
  title     = {Auto-Encoding Variational Bayes},
  author    = {Kingma, Diederik P. and Welling, Max},
  booktitle = {2nd International Conference on Learning Representations, {ICLR} 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings},
  year      = {2014},
  eprint    = {1312.6114},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  url       = {https://arxiv.org/abs/1312.6114}
}
```

**Table row**
- regulariser · per-sample KL(q(z|x) || p(z)) inside the ELBO
- sliced/full · —
- prior · N(0,I)
- nuisance · not treated
- dimensional bottleneck · latent dimension fixed by design
- information bottleneck · implicit, through the KL weight (explicit only in beta-VAE)
- identifiability · none claimed
- objective · autoencoder
- data · MNIST and Frey Faces (natural images)
