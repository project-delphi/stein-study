# zhao2019infovae

**Claim.** The ELBO can give poor amortised posteriors and lets latents be ignored under flexible decoders. InfoVAE adds a weight on mutual information between data and latents and a separate weight on a divergence between the aggregate posterior q(z) and the prior. This improves inference and latent use.

**Method.** Objective: reconstruction, minus (1-alpha) per-example KL, minus (alpha+lambda-1) times a divergence D(q(z)||p(z)). alpha=0, lambda=1 gives the ELBO; alpha+lambda=1 gives beta-VAE; adversarial D gives AAE-like models. D is estimated by MMD (recommended), a Stein variational gradient variant (yes, a Stein variant appears; it is SVGD-style, not KSD), or an adversarial JS estimate. Full (unsliced) batch estimates; paper text names no specific kernel ("any positive definite kernel"). Prior is N(0,I). Datasets: toy two-point mixture, binarised MNIST (subsets), CIFAR qualitative only. Metrics: MMD to prior, posterior accuracy, log-likelihood, SVM semi-supervised error.

**Overlap with this study.** Direct precedent for the VAE tier: aggregate-posterior matching (MMD-VAE ~ WAE-MMD), beta-VAE as a special case, and a Stein variant. Dimensional vs information bottleneck is explicit (alpha trades MI against KL).

**Open gap.** No KSD-to-prior (score-based, IMQ) regulariser, no non-Gaussian prior, no nuisance or leakage measurement, no identifiability analysis, no JEPA or temporal objective, no rendered data with known factors such as 3DShapes.

Verification: DOI and AAAI 2019 venue confirmed by Crossref; arXiv record confirmed (arXiv title differs: "Information Maximizing Variational Autoencoders"). Method details came from an HTML rendering summarised by a tool, not read from the PDF directly; recheck the kernel and Stein details before citing.

## BibTeX
```
@article{Zhao_2019,
  title={InfoVAE: Balancing Learning and Inference in Variational Autoencoders},
  volume={33},
  ISSN={2159-5399},
  url={http://dx.doi.org/10.1609/aaai.v33i01.33015885},
  DOI={10.1609/aaai.v33i01.33015885},
  number={01},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  publisher={Association for the Advancement of Artificial Intelligence (AAAI)},
  author={Zhao, Shengjia and Song, Jiaming and Ermon, Stefano},
  year={2019},
  month=July,
  pages={5885--5892}
}
```

## Table row
- regulariser: MMD (main), SVGD-style Stein, adversarial JS on q(z) vs prior, plus MI weight
- sliced/full: full
- prior: N(0,I)
- nuisance: not treated
- dimensional bottleneck: latent dimension fixed, not studied
- information bottleneck: yes, via alpha (MI) and (1-alpha) KL weight
- identifiability: not addressed
- objective: autoencoder (VAE)
- data: synthetic toy, binarised MNIST (natural-ish images), CIFAR qualitative
