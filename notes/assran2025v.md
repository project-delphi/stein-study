# assran2025v: V-JEPA 2 (Assran et al., 2025, arXiv:2506.09985)

Status: VERIFIED (title, authors, year from arXiv API/BibTeX; arXiv preprint, no venue confirmed). Method details from the arXiv HTML version.

## Claim
Pre-training a masked joint-embedding predictive model on over 1M hours of internet video gives strong motion understanding and action anticipation. It also supports video QA once aligned with an LLM. Post-training an action-conditioned predictor (V-JEPA 2-AC) on under 62 hours of unlabeled robot video enables zero-shot planning on Franka arms.

## Method
Collapse is prevented only by an EMA target encoder plus stop-gradient on the target branch. No variance, covariance, or distributional regulariser is described. The loss is L1 between the predictor output and the EMA-encoder features of the masked patches. V-JEPA 2-AC freezes the encoder. A ~300M block-causal transformer predicts next-frame embeddings from past frames, actions and end-effector states. It is trained with a teacher-forcing L1 loss plus a two-step rollout L1 loss. Planning minimises L1 distance to a goal-image embedding using the cross-entropy method in a model-predictive loop. Data: VideoMix22M (SSv2, Kinetics, HowTo100M, YT-Temporal-1B, ImageNet) for pretraining, and Droid robot video for the AC stage.

## Overlap with this study
- It is the large-scale reference for the EMA+stop-grad baseline.
- The AC stage is a temporal JEPA: it predicts target-frame embeddings from context frames in latent space. Ours uses an EMA target too.
- It uses natural data, but no known factors.
- It has no Stein, KSD or MMD regularisation, no injected noise, no nuisance treatment, no identifiability analysis, and no explicit dimensional or information bottleneck.

## Open gap
It never compares EMA with distributional regularisers (SIGReg, VICReg, KSD, MMD) at matched settings. It never varies embedding dimension or noise σ, and it never measures nuisance leakage or linear recovery of known factors. The study's controlled paired-view setup on 3DShapes, CIFAR-10 and Moving-MNIST covers all of these.

## BibTeX
(The arXiv export splits one author in two, "Mojtaba and Komeili". It is corrected here to the single author Mojtaba Komeili. This is the only change.)

```bibtex
@misc{assran2025vjepa2selfsupervisedvideo,
      title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
      author={Mido Assran and Adrien Bardes and David Fan and Quentin Garrido and Russell Howes and Mojtaba Komeili and Matthew Muckley and Ammar Rizvi and Claire Roberts and Koustuv Sinha and Artem Zholus and Sergio Arnaud and Abha Gejji and Ada Martin and Francois Robert Hogan and Daniel Dugas and Piotr Bojanowski and Vasil Khalidov and Patrick Labatut and Francisco Massa and Marc Szafraniec and Kapil Krishnakumar and Yong Li and Xiaodong Ma and Sarath Chandar and Franziska Meier and Yann LeCun and Michael Rabbat and Nicolas Ballas},
      year={2025},
      eprint={2506.09985},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.09985},
}
```

**Table row**
- regulariser: EMA target + stop-gradient (no distributional regulariser)
- sliced/full: —
- prior: —
- nuisance (treated? how): not treated
- dimensional bottleneck: no
- information bottleneck: no
- identifiability: not studied
- objective: JEPA (masked, L1); temporal JEPA in the action-conditioned stage
- data: natural (web video, ImageNet, Droid robot video)
