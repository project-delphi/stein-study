# Bardes et al. 2024, V-JEPA (TMLR)

Status: VERIFIED. Title, authors and arXiv id from the arXiv BibTeX. Venue (Transactions on Machine Learning Research, 2024) from the ML Anthology record. I did not open the OpenReview page itself.

## Claim
Feature prediction alone, with no pixel reconstruction, text, negatives or pretrained image encoder, yields frozen video representations that serve both motion and appearance tasks. It trains faster than pixel prediction under frozen probing.

## Method
Collapse is prevented only by an EMA target encoder with stop-gradient, plus an L1 loss. No variance, covariance or distributional regulariser. A narrow ViT predictor (12 blocks, width 384) regresses target-patch embeddings from context embeddings and mask tokens. Multi-block spatial masks are repeated across all frames: short-range (8 blocks, ~15% of each frame) and long-range (2 blocks, ~70%), about 90% masked overall. Pretraining uses VideoMix2M (~2M videos from HowTo100M, Kinetics-400/600/700, Something-Something-v2). Evaluation is by frozen attentive probing.

## Overlap with this study
- Is the EMA+stop-grad baseline arm, and a temporal-JEPA reference (masked spatiotemporal prediction in embedding space).
- No Stein/KSD/MMD regularisation, no explicit nuisance model, no injected noise.
- Dimensional bottleneck only implicit (predictor width 384).
- No identifiability analysis.
- Natural video, no known factors.

## Open gap
No test of whether EMA alone avoids nuisance leakage or retains linearly identifiable content. No comparison against distributional regularisers under controlled d versus σ bottlenecks. No data with known generative factors.

## BibTeX
```
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Adrien Bardes and Quentin Garrido and Jean Ponce and Xinlei Chen and Michael Rabbat and Yann LeCun and Mahmoud Assran and Nicolas Ballas},
  journal={Transactions on Machine Learning Research},
  year={2024},
  eprint={2404.08471},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2404.08471}
}
```
(arXiv fields from arxiv.org/bibtex; journal field added from the ML Anthology record.)

**Table row**
- regulariser: EMA target + stop-grad (L1 loss); no distributional term
- sliced/full: —
- prior: —
- nuisance (treated? how): not treated explicitly; only the 90% masking acts implicitly
- dimensional bottleneck: implicit (narrow predictor, width 384)
- information bottleneck: — (masking ratio only, not formalised)
- identifiability: not studied
- objective: temporal/spatiotemporal JEPA (masked feature prediction)
- data: natural (VideoMix2M)
