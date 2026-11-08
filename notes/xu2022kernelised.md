# xu2022kernelised

Status: VERIFIED. The title, authors and year match arXiv:2206.00149. The NeurIPS 2022 venue is confirmed by the proceedings PDF at proceedings.nips.cc and the neurips.cc poster page.

**Claim**
NP-KSD is a kernelised Stein discrepancy test for implicit generators, which have no explicit density. The Stein operator is estimated from generator samples. The generator sample can be as large as desired, while the observed sample size n is fixed. The authors report better power than existing approaches such as MMDAgg.

**Method**
- The paper works with univariate conditionals of each coordinate. Each is conditioned on the other coordinates or on a low-dimensional summary statistic of them.
- The conditional score is estimated from N generator samples by score matching. This gives a score-based Stein operator, averaged over all coordinates or over B randomly chosen coordinates.
- Summary statistics studied: the full conditioning vector, the mean of the other coordinates, and a fitted Gaussian conditional.
- The statistic is a Gaussian-kernel V-statistic on the observed data.
- The null distribution is simulated by drawing fresh size-n sets from the generator. It is not a wild bootstrap. The test rejects above the (1-alpha) quantile.
- Full, not sliced.

**Overlap with this study**
- It gives a Stein-type fit metric of samples against a target, and could score embeddings against an intended prior.
- It is not a regulariser, and it does not treat JEPA, temporal JEPA, autoencoder latents, nuisance, bottlenecks or identifiability.
- Its null is conditional, relative to the chosen summary statistic.
- Real data are MNIST and CIFAR-10 generators, with no known factors.

**Open gap**
- No use as a training loss or as a probe of JEPA embeddings.
- No IMQ kernel and no sliced variant.
- Score matching in high dimensions is costly and depends on the summary statistic.
- No comparison with SIGReg, VICReg or MMD regularisers.

**BibTeX**
```
@inproceedings{xu2022kernelised,
  title     = {A Kernelised Stein Statistic for Assessing Implicit Generative Models},
  author    = {Xu, Wenkai and Reinert, Gesine},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  eprint    = {2206.00149},
  archivePrefix = {arXiv},
  url       = {https://proceedings.nips.cc/paper_files/paper/2022/file/2fb462e23667ad5e6471a4e9af8e4774-Paper-Conference.pdf}
}
```
Volume, pages and editors are not added because I did not read them.

**Table row**
- regulariser: — (a test statistic, NP-KSD; not used as a loss)
- sliced/full: full (coordinate-wise conditionals, no slicing)
- prior: — (target is the generator's implicit distribution)
- nuisance: — (not treated)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: other (generative-model goodness-of-fit testing)
- data: synthetic (Gaussian and mixture) and natural (MNIST, CIFAR-10, CIFAR-10.1)
