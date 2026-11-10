# Related theory for study 2 (T1–T4)

Built from the eight verified notes added in P0 (one per paper, `notes/<citekey>.md`), plus study 1's
`vonkugelgen2021self` and `gorham2017measuring`. Where a note says only the abstract or a keyword scan was
read, "not found" means not found in what was read, and the paper must be cited that way.

**Tian, Chen & Ganguli 2021** (`tian2021understanding`). Linear-network dynamics of non-contrastive SSL
with a learned predictor (BYOL/SimSiam). Its closest relative to T1 is a per-eigenmode growth condition
under weight decay (directions below a threshold die), which is a threshold on modes, not on a nuisance
variable. Its eigenspace-alignment result for the learned predictor is related to our Lemma 1(ii) and
Proposition 1 (the predictor's state in a direction decides whether that direction can be filled). It has
no content/nuisance split, no α, no distributional regulariser. **T1–T4 are not contained**; we cite it for
linear-predictor dynamics and the role of the predictor's eigenbasis.

**Jing et al. 2022** (`jing2022understanding`). Dimensional collapse in contrastive SSL, with a linear
model in which a direction collapses when augmentation variance exceeds signal variance. That is a
threshold in spirit, but in a different quantity (augmentation strength, no λ, no regulariser target).
The note's section pointers were not re-read; cite the paper's main claim only. **Not contained.**

**Simon et al. 2023** (`simon2023stepwise`). Closed-form, stepwise eigenmode learning for a Barlow-Twins-type
loss in the linear regime; VICReg appears only empirically. Read from the abstract only. Related to our
reduction of the objective to an eigenvalue problem (Theorems 1–2), but it concerns learning dynamics,
not a nuisance threshold. **Not found.**

**Balestriero & LeCun 2022** (`balestriero2022contrastive`). VICReg, SimCLR and Barlow Twins as spectral
embedding methods, with closed-form linear-regime optima. Closest to our VICReg analysis and to the
eigenvalue reduction, but it has no nuisance resampling, no threshold in λ, no kernel regulariser and no
predictor. Abstract plus keyword scan only. **Not found**; we cite it as the spectral view that our
reduction shares.

**Shwartz-Ziv et al. 2023** (`shwartzziv2023information`). An information-theoretic reading of VICReg.
Bears on the information- vs dimensional-bottleneck framing of study 1's H2 and T3, not on the threshold
itself. Abstract only. **Not found.**

**Comon 1994** (`comon1994independent`). Theorem 11: independent, non-degenerate sources with at most one
Gaussian, orthogonal mixing; independent outputs ⇔ scaled permutation (via a variant of Darmois's theorem).
The paper also notes that without non-Gaussianity any orthogonal mixing is allowed. **T4(ii) is Comon's
theorem**, and the rotation-invariance half of T4 is implicit in it; our contribution is only the
application to distributional regularisers (which ones are O(d)-invariant, Prop. 3) and the consequence
for the choice of metric (MCC vs orthogonal-fit R²).

**Zimmermann et al. 2021** (`zimmermann2021contrastive`). Contrastive learning identifies latents up to
an orthogonal map (vMF on the sphere) or up to a signed permutation (Lᵖ, p ≠ 2, on a convex body),
measured with R² and MCC. **This is a close relative of T4**: the same orthogonal-versus-permutation
dichotomy and the same pair of metrics, with the asymmetry coming from the positive-pair conditional
rather than from a Gaussian versus non-Gaussian target prior. T4 is new only in where the asymmetry sits
(the regulariser's target law in a JEPA) and must be presented as an analogue of their result.

**Shah et al. 2020** (`shah2020pitfalls`). Extreme simplicity bias in supervised classifiers. It motivates
the open question left by Theorem 1 cor. (d) — why training fills spare directions with nuisance when extra
content features would cost nothing — but it does not study self-supervised embeddings and gives no
evidence on that question. Cited as motivation only.

**Study-1 notes.** `vonkugelgen2021self` (content isolated from style by alignment plus entropy with
resampled style) is the identifiability counterpart of T1 at the level of blocks: it assumes the encoder
dimension equals the content dimension, which is exactly the d ≤ k case where Theorem 1(a) gives no
nuisance; T1 adds what happens when d > k under a moment or kernel penalty. `gorham2017measuring` gives
conditions under which KSD vanishes only at the target; T4(ii) needs that property for the matched
priors, and we do not verify it for the surrogate scores used.

**Novelty claims, as far as these notes show.** Theorem 1 (the λ threshold, its α(2−α) form with a learned
predictor, the m < d − k collapse), Theorem 2 (exact scale reduction for any Σ-dependent regulariser) and
the per-regulariser thresholds, Proposition 1 (empty-direction traps) and the T3 computations were not
found in the eight papers. These are elementary consequences of standard matrix inequalities, and the
literature search was limited to eight papers, so the paper should claim them as derivations and tests in
this setting, not as broad theoretical novelty. T4 is Comon's theorem applied, with Zimmermann et al. as
the closest prior statement of the orthogonal-versus-permutation dichotomy.
