# assran2023self — I-JEPA (VERIFIED)

Verified: arXiv 2301.08243 (title, 8 authors, 2023) and CVF open-access page (CVPR 2023, pp. 15619-15629; DOI 10.1109/CVPR52729.2023.01499). Note: the arXiv comment field says ICCV, which looks like a metadata slip; the CVF page confirms CVPR.

## Claim
Semantic image features can be learned without hand-crafted augmentations by predicting, in representation space, the embeddings of several target blocks from a single context block of the same image. The masking strategy (large-scale targets, spatially informative context) is what drives semantic quality. It scales well with ViTs.

## Method
Context encoder plus narrow ViT predictor (width 384; depth 6 for ViT-B up to 16 for ViT-G) regress target-encoder patch embeddings with mean squared L2. Collapse prevention is architectural: the target encoder is an EMA of the context encoder (momentum 0.996 rising to 1), and no gradient flows to it. The paper does not use the term "stop-gradient" (via ar5iv text). There is no explicit distributional regulariser. Masking: M=4 target blocks (scale 0.15-0.2, aspect 0.75-1.5); one context block (scale 0.85-1.0) with target overlap removed. Data: ImageNet-1K (ImageNet-22K for scaling).

## Overlap with this study
Serves as the EMA+stop-grad baseline for JEPA collapse prevention. Nuisance is not modelled; spatial masking replaces paired views. No Stein/KSD/MMD, no bottleneck analysis, no identifiability claims, no known-factor data.

## Open gap
No distributional (SIGReg/KSD/MMD) alternative to EMA, no dimension-vs-noise bottleneck analysis, no nuisance-leakage or linear-identifiability metrics, no controlled-factor data.

## BibTeX
```bibtex
@InProceedings{Assran_2023_CVPR,
    author    = {Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
    title     = {Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {15619-15629}
}
```

**Table row**
- regulariser: EMA + stop-grad (target encoder is EMA, no gradient to it)
- sliced/full: —
- prior: —
- nuisance: not treated (masking only; no augmentations)
- dimensional bottleneck: no (narrow predictor only, not studied)
- information bottleneck: no
- identifiability: not addressed
- objective: JEPA
- data: natural (ImageNet-1K)
