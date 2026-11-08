# tolstikhin2018wasserstein: Wasserstein Auto-Encoders (ICLR 2018) — VERIFIED

Verified: arXiv:1711.01558 record (title, four authors); ICLR 2018 (Vancouver) via MPI-IS/ELLIS publication record, OpenReview id HkL7n1-0b (listed in search results, page not opened). Method details from the ar5iv text.

## Claim
Auto-encoders can be derived from optimal transport: minimise reconstruction cost plus a penalty λ·D(Q_Z, P_Z) that matches the aggregate posterior to the prior, rather than a per-sample KL as in a VAE. This gives sharper samples while keeping stable encoder-decoder training.

## Method
Penalty on the aggregate posterior Q_Z = E_X[Q(Z|X)] vs P_Z, squared-error cost, λ = 10. WAE-GAN: adversarial latent discriminator (Jensen-Shannon style). WAE-MMD: MMD with a characteristic kernel, minibatch U-statistic. RBF failed to penalise outliers; they used inverse multiquadratic k = C/(C+‖x−y‖²), C = 2·d_z·σ_z², and a sum of IMQs at scales {0.1…10} in the sweep. Prior: isotropic Gaussian N(0, σ_z² I). Deterministic encoders. Data: MNIST (d_z = 8), CelebA (d_z = 64). Full (unsliced) kernel.

## Overlap with this study
Direct baseline: distribution-level regularisation of autoencoder latents with an IMQ-kernel discrepancy, the same family as the KSD-to-prior arm. No nuisance handling, JEPA, or temporal objective. Notes d_z above intrinsic dimension makes Gaussian matching impossible (dimensional bottleneck).

## Open gap
No identifiability claim, no known-factor data, no nuisance-leakage measurement, no Stein discrepancy or non-Gaussian prior, no comparison with JEPA.

## BibTeX
```bibtex
@inproceedings{tolstikhin2018wasserstein,
  title     = {Wasserstein Auto-Encoders},
  author    = {Tolstikhin, Ilya and Bousquet, Olivier and Gelly, Sylvain and Schoelkopf, Bernhard},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  url       = {https://openreview.net/forum?id=HkL7n1-0b},
  note      = {arXiv:1711.01558}
}
```

## Table row
- regulariser: MMD (IMQ) or GAN discriminator on Q_Z vs P_Z
- sliced/full: full
- prior: N(0, σ_z² I)
- nuisance: — (not treated)
- dimensional bottleneck: yes (d_z = 8 / 64; d_z above intrinsic dim. is problematic)
- information bottleneck: — (distribution matching only, no explicit IB)
- identifiability: — (not claimed)
- objective: autoencoder
- data: natural (MNIST, CelebA)
