# xu2021generalised

Verified: arXiv v1 abstract page (title/author/date 23 Jun 2021), arXiv BibTeX endpoint, and PMLR page. Venue: AISTATS 2022 (PMLR 151), where the paper was retitled "Standardisation-function Kernel Stein Discrepancy (Sf-KSD): A Unifying View on Kernel Stein Discrepancy Tests for Goodness-of-fit". Only the abstracts were read; the PDF text could not be extracted in this environment, so operator details are not confirmed.

## Claim
Stein operators are not unique, and the choice affects the power of KSD goodness-of-fit tests as much as the kernel does. GKSD (later Sf-KSD) is a framework for comparing and interpreting different Stein operators. It shows how existing operators and their tests arise as special cases. It is also a guide for new tests on truncated distributions and compositional data.

## Method
Goodness-of-fit testing of unnormalised densities with a KSD whose Stein operator is a design choice. The abstract does not name the individual operators it unifies. The v2 abstract ties the framework to the "standardisation" technique from Stein's method for normal approximation. Reported results: type-I error is controlled, and power exceeds existing tests, including an MMD test.

## Overlap with this study
Indirect only. It supports treating the Stein operator as a free design choice alongside the kernel when using KSD toward Gaussian or non-Gaussian priors. It does not cover JEPA, latents, nuisance, bottlenecks or identifiability. It is a hypothesis test, not a training regulariser.

## Open gap
No use of KSD as a differentiable regulariser. No sliced or conditional variants are confirmed from what I read. No image data. The study could compare Stein operator choices (and IMQ versus other kernels) as regularisers under JEPA bottlenecks.

## BibTeX
```bibtex
@InProceedings{pmlr-v151-xu22b,
  title     = {Standardisation-function Kernel Stein Discrepancy: A Unifying View on Kernel Stein Discrepancy Tests for Goodness-of-fit},
  author    = {Xu, Wenkai},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  pages     = {1575--1597},
  year      = {2022},
  volume    = {151},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v151/xu22b.html},
  eprint    = {2106.12105},
  archivePrefix = {arXiv}
}

@misc{xu2022standardisationfunctionkernelsteindiscrepancy,
  title         = {Standardisation-function Kernel Stein Discrepancy: A Unifying View on Kernel Stein Discrepancy Tests for Goodness-of-fit},
  author        = {Wenkai Xu},
  year          = {2022},
  eprint        = {2106.12105},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  url           = {https://arxiv.org/abs/2106.12105}
}
```
(The PMLR fields are from the page metadata; booktitle/volume/publisher were read there, year from publication date 2022/05/03.)

**Table row**
- regulariser · KSD (test statistic, generalised Stein operator)
- sliced/full · —
- prior · — (target is the null density of the test)
- nuisance · —
- dimensional bottleneck · —
- information bottleneck · —
- identifiability · —
- objective · other (goodness-of-fit test)
- data · —
