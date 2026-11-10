# Shah et al. 2020, The pitfalls of simplicity bias in neural networks

Status: VERIFIED (arXiv record 2006.07710 v2: title, five authors, year 2020; arXiv comment "NeurIPS 2020"; the proceedings paper page at papers.nips.cc/neurips.cc was found in search, but volume and pages were not seen, so they are omitted from the BibTeX).

## Claim
SGD-trained networks can show an extreme simplicity bias. Given several features that each predict the label perfectly, they use only the simplest one and stay invariant to the more complex ones. Simplicity here has a precise meaning: the minimum number of linear pieces in the decision boundary. The authors argue this explains fragility to distribution shift and small perturbations, and it can hurt in-distribution generalisation. Ensembles and adversarial training do not fix it.

## Method
Supervised binary classification on constructed datasets: LMS-k (one linear coordinate plus many k-slab coordinates), a noisy-linear variant, MS-(5,7), MS-5, and MNIST-CIFAR (MNIST digit stacked on a CIFAR-10 image; MNIST is the simple block). Models are fully connected nets (one-hidden-layer ReLU nets, with a proof for the stylised LSN dataset) and, on MNIST-CIFAR, MobileNetV2, GoogLeNet, ResNet50 and DenseNet121 (Sections 3.1, 4.1, 4.2, read in fetched text). Reliance on a feature is measured by randomising that feature and checking logits and AUC. No regulariser, kernel or prior is involved.

## Overlap with this study
- Bears on heuristic T1d only as a candidate explanation, and only indirectly. The paper shows that in supervised discriminative training, when a linearly separable feature (low piece count) and more complex but equally predictive features are both available, the network relies on the simple one. This supports "networks prefer simple, e.g. linearly decodable, features".
- Caveats for transfer. (1) The paper is supervised with a label loss. It does not study self-supervised, JEPA, autoencoder or embedding-dimension settings (nothing on these appears in the sections read). (2) Its "simple" means few decision-boundary pieces for a label, not low-frequency or linearly decodable content in a latent. (3) The simple feature suffices alone for 100% accuracy, so extra features are redundant for the loss. In a JEPA with spare dimensions, nuisance filling is a different phenomenon: the loss does not reward it, so it needs its own mechanism (for example, nuisance is simply the easiest high-variance signal to fit). This paper therefore cannot confirm or refute T1d, and I did not find any statement in it about dimension counts, nuisance, KSD/MMD/Stein, identifiability or real data with known factors.
- It is a reasonable citation for a simplicity-bias hypothesis, offered as motivation and not as evidence.

## Open gap
No test of whether simplicity bias operates in self-supervised or predictive-embedding objectives, or whether spare latent dimensions are filled by simple nuisance versus complex content. The study could build such a test, for example a 3DShapes JEPA where factors vary in complexity (linear vs periodic/high-frequency functions of the content), checking which dimensions the encoder fills when capacity is spare, with and without a Stein/MMD regulariser.

## BibTeX
```bibtex
@inproceedings{shah2020pitfalls,
      title={The Pitfalls of Simplicity Bias in Neural Networks},
      author={Harshay Shah and Kaustav Tamuly and Aditi Raghunathan and Prateek Jain and Praneeth Netrapalli},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2020},
      eprint={2006.07710},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.07710}
}
```
(Fields from the arXiv BibTeX record; booktitle and the NeurIPS venue from the arXiv comment and the NeurIPS 2020 proceedings page found by search. Volume and pages not confirmed.)
