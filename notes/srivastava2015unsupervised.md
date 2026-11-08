# Srivastava, Mansimov & Salakhutdinov 2015, Unsupervised Learning of Video Representations using LSTMs

Status: VERIFIED (PMLR page v37/srivastava15 and arXiv 1502.04681; PMLR spells the surname "Salakhudinov", arXiv "Salakhutdinov").

**Claim**
LSTM encoder-decoder networks learn video representations without labels, by reconstructing the input, predicting future frames, or both (composite model). Features help action recognition, most when labelled data is scarce.

**Method**
Encoder LSTM compresses frames to a fixed-length state. Decoders reconstruct or predict, trained with cross-entropy on logistic outputs. No distributional regulariser, kernel or prior. Moving MNIST: 20-frame videos, two random training digits in a 64×64 frame, random start and velocity, wall bounces, overlaps allowed. 10 frames in, 10 reconstructed, 10 predicted. Also tested on one and three digits.

**Overlap with this study**
Origin of Moving MNIST, a candidate temporal-tier dataset (label = content, motion = nuisance). Future prediction is a pixel-space precursor of temporal JEPA. No Stein/KSD/MMD, identifiability or nuisance treatment.

**Open gap**
No latent-space prediction, EMA target or latent regularisation. No test of whether digit identity is recoverable or motion is discarded.

**BibTeX**
```bibtex
@inproceedings{srivastava2015unsupervised,
  title     = {Unsupervised Learning of Video Representations using {LSTM}s},
  author    = {Srivastava, Nitish and Mansimov, Elman and Salakhutdinov, Ruslan},
  booktitle = {Proceedings of the 32nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {37},
  pages     = {843--852},
  year      = {2015},
  publisher = {PMLR},
  eprint    = {1502.04681},
  archivePrefix = {arXiv}
}
```

**Table row**
- regulariser: —
- sliced/full: —
- prior: —
- nuisance: not treated (motion is modelled, not removed)
- dimensional bottleneck: yes, fixed-size LSTM state (2048 units in the MNIST model)
- information bottleneck: —
- identifiability: —
- objective: autoencoder plus future prediction (pixel space, other than JEPA)
- data: synthetic (Moving MNIST), plus natural video (UCF-101, HMDB-51)
