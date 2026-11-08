VERIFIED (venue confirmed during P0 synthesis): title, authors and year from arXiv 2002.07017; ICLR 2020 and the five-author list confirmed from the ML Anthology record https://mlanthology.org/iclr/2020/federici2020iclr-learning/ (venue: International Conference on Learning Representations, ICLR 2020). The reviewer had initially left the venue unconfirmed because OpenReview and iclr.cc could not be fetched.

# federici2020learning

**Claim**
Two views of the same entity share the task-relevant content. A representation that is predictive of the other view is sufficient, and discarding what the views do not share (superfluous, nuisance-like information) makes it minimal. This gives an unsupervised information bottleneck without labels.

**Method**
Objective: for each view, maximise I(z1; v2) while penalising I(z1; v1 | v2). The penalty is bounded by a symmetrised KL between the two Gaussian posteriors p(z1|v1) and p(z2|v2), with a trade-off weight β. The MI term uses a Jensen-Shannon or InfoNCE critic (JS was better). Stochastic encoders; no kernel; the prior is replaced by the other view's posterior. Single-view extension via augmentation. Datasets: Sketchy (sketch-to-image retrieval), MIR-Flickr (image and tags, low labels), and MNIST with two augmented views (affine, 20% pixel corruption).

**Overlap with this study**
Same paired-view setup: shared content, view-specific nuisance. Its β/posterior noise is an information bottleneck, comparable to our σ² injection. No Stein/KSD/MMD, no JEPA, no dimensional-versus-information comparison, no identifiability result.

**Open gap**
Collapse and distribution matching are not regularised. Dimensional and information bottlenecks are not compared, and nuisance leakage is not measured as R² against known nuisance factors. Sufficiency and minimality are argued, not identifiability.

**BibTeX**
```
@inproceedings{federici2020learning,
  title     = {Learning Robust Representations via Multi-View Information Bottleneck},
  author    = {Marco Federici and Anjan Dutta and Patrick Forr{\'e} and Nate Kushman and Zeynep Akata},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  eprint    = {2002.07017},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2002.07017}
}
```
(The arXiv BibTeX gives only title, authors, year, eprint and url. The booktitle is from the ML Anthology record; the accent in Forré is added from the author's name.)

**Table row**
- regulariser: symmetrised KL between view posteriors (MIB) plus an MI critic
- sliced/full: —
- prior: none; the other view's posterior acts as the target
- nuisance: treated, as non-shared information penalised by the I(z1; v1 | v2) bound
- dimensional bottleneck: no (1024-d, with 2x64 parameters for MNIST)
- information bottleneck: yes (β-weighted)
- identifiability: no formal result; sufficiency and minimality only
- objective: other (multi-view IB, InfoMax-style)
- data: natural (Sketchy, MIR-Flickr) and augmented MNIST
