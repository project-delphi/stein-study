# vonkugelgen2021self — VERIFIED

Verified: arXiv:2106.04619 (title/authors; comment "NeurIPS 2021 camera-ready"); NeurIPS 2021 proceedings PDF at papers.neurips.cc, AIS 34, pp. 16451–16467 (pages per MPI-IS record; page range not re-read from the PDF itself).

## Claim
If two augmented views share a content block and differ only in a random subset of style coordinates, then contrastive-style training (alignment plus entropy maximisation) recovers content up to an invertible map, while style is discarded. Latents may be statistically or causally dependent.

## Method
Generative model x = f(z), z = (c, s); view 2 keeps c and resamples a random subset of style coordinates. Objective L_AlignMaxEnt = E||g(x)-g(x̃)||² − H(g(x)), with encoder output of dimension n_c in (0,1)^{n_c}; InfoNCE approximates it in practice. No kernel, no prior, no sliced/full distinction. Assumptions: f a diffeomorphism (smooth, invertible); p_z smooth and positive a.e.; every style variable lies in some changing subset with positive probability, with smooth conditional positive on an open neighbourhood; global optimum, asymptotic. Result: ĉ = h(c), h invertible (block identifiability; no within-block disentanglement). Empirics: numerical dependent-latent simulations, and Causal3DIdent (rendered 224×224, 7 object classes, 10 latent factors, causal dependencies among object/lighting hues), ResNet18 with InfoNCE, content dim 8, 3 seeds.

## Overlap with this study
Close match to the paired-view setup (shared content, resampled nuisance with probability α) and to linear/invertible identifiability of content on data with known factors. Nuisance = style. The dimension assumption (n_c known) is a dimensional bottleneck. No Stein/KSD/MMD regularisation, no JEPA predictor, no noise injection.

## Open gap
Entropy is estimated through negatives rather than a distributional regulariser (SIGReg, KSD, MMD). The study could test whether those regularisers retain block identifiability when d exceeds n_c or when σ noise acts as an information bottleneck, and measure nuisance leakage as α varies. Temporal JEPA and autoencoders are not covered.

## BibTeX
```bibtex
@inproceedings{vonkugelgen2021self,
  title     = {Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style},
  author    = {von K{\"u}gelgen, Julius and Sharma, Yash and Gresele, Luigi and Brendel, Wieland and Sch{\"o}lkopf, Bernhard and Besserve, Michel and Locatello, Francesco},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {16451--16467},
  year      = {2021},
  url       = {https://papers.neurips.cc/paper_files/paper/2021/file/8929c70f8d710e412d38da624b21c3c8-Paper.pdf},
  note      = {arXiv:2106.04619}
}
```

**Table row**
- regulariser: — (alignment + entropy / InfoNCE)
- sliced/full: —
- prior: —
- nuisance (treated? how): yes; style resampled in random subsets across views, discarded by invariance
- dimensional bottleneck: yes; encoder output dim = n_c
- information bottleneck: —
- identifiability: yes; block identifiability of content up to invertible map
- objective: other (contrastive / InfoNCE, multi-view)
- data: synthetic and rendered (Causal3DIdent)
