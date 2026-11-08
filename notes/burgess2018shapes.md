# burgess2018shapes — 3D Shapes Dataset (Burgess & Kim, 2018)

Status: VERIFIED against the official README, github.com/google-deepmind/3d-shapes (the README's BibTeX uses the older deepmind/3dshapes-dataset URL). No paper exists for the dataset itself.

**Claim**
A procedurally generated image set built from six independent ground-truth factors in a full factorial grid. It was released for evaluating disentanglement; Kim & Mnih (ICML 2018) used it.

**Method**
N = 480,000 images, 64×64×3, each factor combination once. Factors: floor hue 10, wall hue 10, object hue 10 (linear in [0,1]), scale 8 (linear in [0,1]), shape 4 (0–3), orientation 15 (linear in [-30,30]). HDF5 fields: `images` (480000×64×64×3, uint8), `labels` (480000×6, float64), stored in fixed order (orientation varies fastest). Rendered with the GQN MuJoCo environment.

**Overlap with this study**
Core dataset: real-label-free ground truth. Content = shape, scale, object hue; nuisance = floor hue, wall hue, orientation. Enables identifiability and nuisance tests. No regularisation in the dataset itself.

**Open gap**
No temporal structure or nuisance split is defined; the study must supply the content/nuisance partition and any temporal pairing. Rendered, not natural.

**BibTeX**
```
@misc{3dshapes18,
  title={3D Shapes Dataset},
  author={Burgess, Chris and Kim, Hyunjik},
  howpublished={https://github.com/deepmind/3dshapes-dataset/},
  year={2018}
}
```
(Verbatim from the README. Intended citekey here: burgess2018shapes.)

**Table row**
- regulariser: —
- sliced/full: —
- prior: —
- nuisance: — (dataset provides labelled nuisance factors: floor hue, wall hue, orientation)
- dimensional bottleneck: —
- information bottleneck: —
- identifiability: — (ground-truth factors enable testing)
- objective: other (dataset)
- data: rendered (synthetic, MuJoCo)
