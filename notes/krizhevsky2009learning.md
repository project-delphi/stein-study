# krizhevsky2009learning

**Claim.** CIFAR-10 is a labelled subset of the 80 million tiny images collection: 60,000 32x32 colour images, 10 mutually exclusive classes, 6,000 per class, split 50,000 train / 10,000 test. The official page asks users to cite the tech report (Chapter 3 describes collection).

**Method.** Dataset only; nothing is regularised. Images are labelled by class.

**Overlap with this study.** Natural-image tier. Two augmentations per image; augmentation parameters are the known nuisance; class label is the shared-signal probe.

**Open gap.** Nuisance factors are not annotated; only the class is. Augmentation parameters must be logged by the study. Institution is not stated on the page, so it is omitted below.

**BibTeX**
```bibtex
@techreport{krizhevsky2009learning,
  title  = {Learning Multiple Layers of Features from Tiny Images},
  author = {Krizhevsky, Alex},
  year   = {2009},
  type   = {Technical report},
  url    = {https://www.cs.toronto.edu/~kriz/cifar.html}
}
```

**Table row**
- regulariser: —
- sliced/full: —
- prior: —
- nuisance: augmentation parameters, known and logged by the study (not annotated in the dataset)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: —
- objective: —
- data: natural
