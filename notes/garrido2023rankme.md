# garrido2023rankme (VERIFIED: PMLR v202 page and arXiv 2210.02885 record)

## Claim
The effective rank of a joint-embedding SSL representation, computed on unlabelled data, predicts linear-probe accuracy well enough to choose hyperparameters with little loss versus label-based selection. Low rank signals dimensional collapse; no tuning is needed.

## Method
Measured, not regularised. Stack N embeddings, take singular values, normalise to p_k = sigma_k / sum of sigma (small epsilon added), and RankMe = exp(-sum p_k log p_k), the exponentiated Shannon entropy. Full spectrum, no kernel or prior. Tested on hundreds of JE-SSL runs (VICReg, SimCLR, etc.) across hyperparameters and downstream datasets.

## Overlap with this study
Direct source of the effective-rank metric used on embeddings. Covers JE-SSL on natural images; no Stein/KSD/MMD regularisers, no nuisance, no identifiability, no explicit dimensional vs noise (information) bottleneck.

## Open gap
Rank is not tested under controlled d or sigma-noise bottlenecks, nor across distributional regularisers; no known-factor ground truth, so rank vs factor recovery is untested.

## BibTeX
```bibtex
@InProceedings{garrido23a,
  title     = {RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank},
  author    = {Garrido, Quentin and Balestriero, Randall and Najman, Laurent and LeCun, Yann},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages     = {10929--10974},
  year      = {2023},
  volume    = {202},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

## Table row
- regulariser: — (metric only)
- sliced/full: full (all singular values)
- prior: —
- nuisance: not treated
- dimensional bottleneck: — (rank collapse diagnosed, d not varied as a bottleneck)
- information bottleneck: —
- identifiability: —
- objective: other (JE-SSL, e.g. VICReg/SimCLR; evaluation metric)
- data: natural (ImageNet and downstream image sets)
