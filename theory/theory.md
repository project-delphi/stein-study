# A linear theory of nuisance in regularised JEPAs — statements, proofs, status

All numbers below come from `theory/check_t*.py` → `results2/theory_t*.json`. Status labels:
**proved** (complete proof below, confirmed numerically), **corrected and proved**, **numerical**
(an exact reduction is proved, the final constant is computed numerically), **heuristic**, **refuted**.

## 0. Model and units

Content s ∈ ℝᵏ and nuisance n ∈ ℝᵐ are independent, zero mean, identity covariance. Views x = (s, n),
x′ = (s, n′); each nuisance coordinate is resampled independently with probability α ∈ (0, 1], so
Cov(n, n′) = (1−α)I. Encoder z = As + Bn ∈ ℝᵈ (A ∈ ℝ^{d×k}, B ∈ ℝ^{d×m}), z′ = As + Bn′. Write
P = AAᵀ, Q = BBᵀ, Σ = Cov z = P + Q, C = Cov(z′, z) = P + (1−α)Q = Σ − αQ. Every pair of PSD matrices
(P, Q) with rank P ≤ k, rank Q ≤ m is realisable.

Objective J = c_p·L_pred + λ·R(Σ). The prompt's form has c_p = 1. **The code's `F.mse_loss` averages
over the d coordinates, so c_p = 1/d in code units**; regularisers are averaged over the two views,
which leaves their expectation unchanged. Two predictors: identity (L_pred = E‖z − z′‖²) and a learned
linear W trained jointly without stop-gradient (study 1's non-EMA arms use no stop-gradient; the
stop-gradient variant is not analysed). r := min(m, (d−k)₊).

The **cov arm** (to be implemented in P1) is R_cov(z) = ‖Σ̂ − I‖²_F with Σ̂ the centred, unbiased batch
covariance. For Gaussian z and batch N, E R_cov = ‖Σ − I‖²_F + (‖Σ‖²_F + (tr Σ)²)/(N − 1) (Wishart
variances); the finite-N term shifts thresholds by < 3 % at N = 256 (`theory_t2.json`: t* = 1.952 vs 2).

## Lemma 1 (invariance cost)

(i) Identity predictor: E‖z − z′‖² = 2α tr Q.
(ii) Learned linear predictor: min_W E‖Wz − z′‖² = 2α tr Q − α² tr(QΣ⁺Q) ≥ α(2−α) tr Q, with equality iff
Q̃ := Σ^{+1/2} Q Σ^{+1/2} is an orthogonal projector (equivalently PΣ⁺Q = 0: content and nuisance
ranges are linearly independent).

*Proof.* (i) z − z′ = B(n − n′), Cov(n − n′) = 2αI. (ii) E‖Wz − z′‖² = tr(WΣWᵀ) − 2tr(WC) + tr Σ is a convex
quadratic in W minimised by any W with WΣ = C, e.g. W = CΣ⁺ (range C ⊆ range Σ), giving tr Σ − tr(CΣ⁺C).
With C = Σ − αQ, ΣΣ⁺Σ = Σ and range Q ⊆ range Σ: tr(CΣ⁺C) = tr Σ − 2α tr Q + α² tr(QΣ⁺Q). Since 0 ⪯ Q ⪯ Σ,
0 ⪯ Q̃ ⪯ Π (projector onto range Σ), so Q̃² ⪯ Q̃ and, with Q = Σ^{1/2}Q̃Σ^{1/2},
tr(QΣ⁺Q) = tr(Σ^{1/2}Q̃²Σ^{1/2}) ≤ tr(Σ^{1/2}Q̃Σ^{1/2}) = tr Q. Equality forces Σ^{1/2}(Q̃ − Q̃²)Σ^{1/2} = 0, i.e.
Q̃ = Q̃² on range Σ. Then P̃ := Π − Q̃ is the complementary projector and P̃Q̃ = 0 ⇔ PΣ⁺Q = 0. ∎

So the invariance cost per unit of nuisance variance is κ = 2α (identity) or κ = α(2−α) (learned linear,
attained when the predictor can separate content from nuisance). For any *nonlinear* predictor and any
feature g of a single resampled factor, the best prediction of g(n′) from g(n) is (1−α)g(n), so the same
α(2−α) per unit variance holds feature-wise (features of j factors cost 1 − (1−α)^{2j}).

## T1 — covariance penalty: threshold. Status: **corrected and proved**

**Theorem 1.** Let R = ‖Σ − I‖²_F, κ as in Lemma 1, α ∈ (0, 1], λ > 0, v* := (1 − c_pκ/(2λ))₊.
(a) d ≤ k: min J = 0; every global minimiser has B = 0 and AAᵀ = I_d.
(b) d > k: every global minimiser has AAᵀ = U Uᵀ for some k-dim orthonormal U; if λ > λ* := c_pκ/2,
BBᵀ = v*·VVᵀ with V an r-dim orthonormal set ⟂ U, and the remaining d − k − r directions have zero
variance; if λ ≤ λ*, B = 0. min J = r[c_pκv* + λ(v* − 1)²] + λ(d − k − r).
Thus λ* = c_p α for the identity predictor (the prompt's sketch, c_p = 1) but **λ* = c_p α(2−α)/2 for the
learned linear predictor**, which is not proportional to α. With m < d − k the d − k − m unfillable
directions are collapsed (variance 0) at every λ.

*Proof.* By Lemma 1, J ≥ J_b := c_pκ tr Q + λR(Σ) (equality for the identity predictor). Fix Σ. Over
decompositions Σ = P + Q with P, Q ⪰ 0, rank P ≤ k, rank Q ≤ m, tr Q = tr Σ − tr P, and P ⪯ Σ gives
λᵢ(P) ≤ λᵢ(Σ) (Weyl), so tr P ≤ Σ_{i≤k} σᵢ (σ₁ ≥ … ≥ σ_d eigenvalues of Σ), attained by the spectral split
P = top-k part, Q = rest — for which Q̃ is a projector, so J = J_b there. Feasibility needs at most m
nonzero σᵢ with i > k. Hence min J = min over σ of f(σ) = Σ_{i≤k} λ(σᵢ−1)² + Σ_{i>k}[c_pκσᵢ + λ(σᵢ−1)²]
s.t. #{i > k : σᵢ > 0} ≤ m. Each summand is strictly convex; the unconstrained minimiser σᵢ = 1 (i ≤ k),
σᵢ = v* (i > k) is sorted, so it solves the ordered problem; with the cardinality constraint, filling a
bottom slot with v* > 0 lowers f by (λ − c_pκ/2)²/λ > 0, so exactly r slots get v* when λ > λ*, none
otherwise. This fixes the spectrum of every minimiser. If v* < 1, any optimal P has tr P = k with
λᵢ(P) ≤ 1, so P has k unit eigenvalues; for a unit eigenvector u of P, uᵀΣu ≥ 1 = σ_max forces u into the
eigenvalue-1 eigenspace of Σ, which has dimension k: P = UUᵀ and Q = Σ − P. For the learned predictor any
minimiser of J attains J_b (else J > min J_b = min J), so the same structure holds. (a) is the case r = 0
with P = I_d. ∎ *Check:* `theory_t1.json` parts A (identity, 5 shapes (d,k,m) incl. d ≤ k and m < d−k,
α ∈ {0.25, 0.5, 1}, λ/λ* ∈ {0.5, 0.8, 1.25, 2, 5}, 2–3 starts; all match to 2e-3), B (fast learned
predictor), C (minibatch training in code units, N = 256: v = 0.466/0.760 vs finite-N solution
0.466/0.760), all PASS.

**Corollaries** (proved from Theorem 1).
(a) The population linear-probe R² of n from z is tr(BᵀΣ⁺B)/m = r/m if v* > 0 and 0 otherwise — a **step**
at λ* (with m ≤ d − k it jumps from 0 to 1) because it is scale-invariant; the nuisance **variance
share** r v*/(k + r v*) is continuous and 0 at λ*.
(b) λ* ∝ κ: ∝ α (identity predictor) or ∝ α(2−α) (learned predictor). At α = 0 nuisance is free and
P + Q = I with any split is optimal (minimiser not unique; check A_alpha0).
(c) d ≤ k: no threshold, no nuisance at any λ.
(d) (Linearity is needed.) If the encoder may also output d − k unit-variance functions of s uncorrelated
with s and each other (available whenever the centred L²(s) has dimension ≥ d; 3DShapes content has
10·8·4 = 320 values, so 319 dimensions), z = (s, g(s)) gives J = 0 < min J_linear with no nuisance;
for the identity predictor every J = 0 solution has E‖z − z′‖² = 0, so z does not depend on n. Check D:
J = 9e-16 vs linear minimum 9.5. So in a nonlinear network leakage above λ* is **not forced** by the
cov objective; which zero-cost solution training finds is outside this theory.

**Proposition 1 (empty-direction traps). Status: heuristic, numerically supported.** If a direction u
is empty (Σu = 0) the predictor receives no gradient along u. Adding variance v along u (content or
nuisance) changes J by v[c_p c(u) − 2λ] + λv² with c(u) = ‖Wu‖² − 2ρ uᵀWu + 1 (ρ = 1 for content,
1 − α for nuisance), so this move is uphill when 2λ < c_p c(u). With Wu = 0 (as for the ridge-optimal or
weight-decayed predictor) c(u) = 1 for both, so the escape point is λ_trap = c_p/2 — **independent of α**
and above λ* = c_pα(2−α)/2 when α < 1. Full local minimality is not proved. Observed: 21/60 joint-W
gradient-descent runs and 6/135 fast-predictor runs ended in such states; every one satisfied
c(u) ≥ 2λ (`theory_t1.json`, B2_joint_W, B_traps). Consequence: the onset seen in training can lie
anywhere between c_pα(2−α)/2 and c_p/2 (for cov), depending on the predictor's state in empty directions.

## T2 — other regularisers and the learned predictor. Status: scaling **proved**; constants **numerical**; quadratic proxy **refuted**; VICReg **heuristic**

**Theorem 2 (scale reduction).** For any regulariser that depends on z only through Σ (every
Gaussian-target arm when z is Gaussian, which it is in this model; VICReg in population), with the
identity or the learned linear predictor, min J = λ·min_σ [t Σ_{i>k} σᵢ + R(σ)] (for orthogonally
invariant R; for general R the minimum is over Σ with the same spectral split), t := c_pκ/λ. Hence the
set of minimisers depends on (α, λ, c_p) only through t, the threshold is
**λ*_reg = c_p κ(α) / t*_reg**, with t*_reg depending on R, d, k, m only, and λ* ∝ α(2−α) for every such arm
(learned predictor).
*Proof.* The argument of Theorem 1 up to "min J = min over σ" used only that R depends on Σ: J ≥ J_b,
for fixed Σ the spectral split minimises tr Q and attains J = J_b. Divide by λ. ∎

**Population forms** (`theory/_lib.py`, each validated against the batch estimator in `stein/losses.py` by
Monte Carlo on three covariances, `theory_t2.json` mc_validation PASS): KSD-IMQ by a one-dimensional
Gamma-integral identity; MMD in closed form; sliced KSD and SIGReg as averages over 4096 fixed directions
of one-dimensional forms; SIGReg's **expected** batch statistic includes the bias term (1 − ψ_t²), since
the Epps–Pulley statistic is multiplied by N (= 256) and is not a U-statistic.

**Thresholds** (t* by bisection on the global eigenvalue minimum, multi-start L-BFGS; confirmed by
gradient descent on (A, B) at λ/λ* ∈ {0.6, 2, 5}, α ∈ {0.5, 1}, all PASS). Code units, k = 6, m = 5:

| arm | λ* (d=12, α=1) | α = 0.5 | α = 0.25 | λ* (d=24, α=1) | t*/t_quad (d=12) |
|---|---|---|---|---|---|
| cov | 0.0427 | 0.032 | 0.0187 | 0.0213 | 0.97 |
| SIGReg | 0.0389 | 0.0292 | 0.017 | 0.0431 | 1.9 |
| KSD | 0.939 | 0.705 | 0.411 | 0.588 | 5.3 |
| sliced KSD | 7.33 | 5.50 | 3.21 | 8.04 | 2.3 |
| MMD | 17.1 | 12.8 | 7.48 | 18.0 | 2.3 |
| VICReg (GD, heuristic) | in (4.6, 10] | — | — | — | — |

d = 6 (= k): no threshold for any arm. The second-order "curvature at I" proxy (t_quad) misses t* by
factors up to 5 (12 at d = 24 for KSD): **the threshold is set by the slope of R at a collapsed
direction (t_loc), not by its curvature at I**; t* agrees with t_loc within 5 % for every arm (no
first-order jump resolved). Above threshold, only cov follows v = 1 − λ*/λ; the kernel arms rise more
slowly near λ* (KSD, d = 12: v = 0.22 at λ = 2λ* vs 0.5) and, at d = 24, exceed 1 (content and nuisance
both inflate to compensate the collapsed directions). Under the trap mechanism (Prop. 1) the escape point
is c_p/t_loc for every α.
*Post-hoc consistency with study 1 (seen data, not a test):* study 1's λ-sensitivity runs (α = 1, d = 12)
show leakage onset between λ = 0.2 and 1 (KSD), 0.01 and 0.05 (SIGReg), 10 and 50 (MMD), 1 and 5 (sliced
KSD), and leakage already 0.17 at λ = 1 (VICReg); the linear values above fall inside the first three
brackets, slightly above the fourth, and well above VICReg's.

**VICReg.** Theorem 2 applies, so a threshold exists and scales with α(2−α). The hinge changes what sits
below it: content alone satisfies the variance term on all d coordinates by spreading its rank-k
covariance as a tight frame (GD: all diagonals = 1.0, content eigenvalues d/k = 2), paying only the small
covariance term, so nuisance is bought late (GD scan: none up to λ = 4.6, share 0.17 at λ = 10, 0.35–0.44 for
λ ≥ 46); above it nuisance variance is not capped at 1. Status heuristic: GD only, and batch fluctuations
of the per-coordinate std are ignored.

## T3 — noise on the predictor input. Status: **heuristic** (fixed-scale numerically confirmed; batch-scaled approximate)

Assumption A-orth: content and nuisance occupy orthogonal principal directions; for batch-scaled noise also
A-diag: principal axes = coordinate axes (noise covariance σ²diag Σ = σ²Σ). With the predictor at its
optimum, per unit variance a direction costs
- batch-scaled (scale detached; fixed points = stationary points of the potential below): content
  κ_s = (σ²/(1+σ²))², nuisance κ_n = w² − 2w(1−α) + 1 with w = (1−α)/(1+σ²) (= 1 at α = 1);
- fixed scale (a true objective): content cσ²/(c+σ²), nuisance v − (1−α)²v²/(v+σ²).
Minimising c_p·(costs) + λR over eigenvalues (`theory_t3.json` part 1):
1. cov (separable): λ* unchanged by σ at α = 1 (0.0427 → 0.0424); content variance falls; nuisance
   variance share rises (0.216 → 0.250 at λ = 1.5λ*); probe R² unchanged.
2. Coupled regularisers: taxing content shrinks it, which raises the value of filling the spare
   directions, so **batch-scaled σ lowers λ***: KSD 0.939 / 0.785 / 0.247 at σ = 0 / 0.5 / 1; SIGReg
   0.039 / 0.037 / 0.025; MMD 17.1 / 15.9 / 10.1; sliced KSD 7.33 / 6.80 / 4.24 (α = 1). Fixed-scale
   σ = 1 lowers KSD's λ* to 0.002 (at low SNR content is no cheaper than nuisance) and the others by
   15–20 %; σ = 0.5 changes them < 7 %.
3. α < 1, batch-scaled: κ_n rises with σ, which pushes the other way for cov (λ* 0.032 → 0.034).
So the linear model predicts the *sign* of study 1's σ effect for the kernel arms via a lower threshold,
but not its size at study 1's KSD level: λ*(σ = 1) = 0.247 > 0.2, so at λ = 0.2 it predicts no leakage,
whereas study 1 observed a large rise. For cov it predicts no change in probe leakage.
*Check (part 2, minibatch training with real noise, α = 1, λ = 2λ*(σ=0)):* fixed-scale σ = 1 matches for
cov and KSD; batch-scaled matches for cov but not for KSD (content 1.12 vs 0.98, nuisance 0.29 vs 0.33;
the trained Σ is far from axis-aligned, off-diagonal norm 0.55 of total), so A-diag fails there; the
direction (content down, nuisance up relative to σ = 0: 1.44/0.22) holds.

## T4 — what a matched prior can identify. Status: (i) **proved**; (ii) **proved by citation** under stated idealisations

**Proposition 3 (i).** For KSD-IMQ, MMD with Gaussian kernels, sliced KSD and SIGReg with random
directions, all with target N(0, I), and any law of z: R(law of Oz) = R(law of z) for every orthogonal O.
*Proof.* The IMQ and Gaussian kernels are functions of ‖x − y‖; the N(0, I) score −x is equivariant, so
the Stein kernel satisfies u(Ox, Oy) = u(x, y) (each of its terms is built from inner products of x, y,
x − y); MMD's target is invariant; sliced statistics average over Haar-distributed directions, and
g ↦ Oᵀg preserves that law. ∎ The prediction loss is invariant under z → Oz, W → OWOᵀ, and the encoder's
last layer is linear, so the whole objective is O(d)-invariant and its minimiser set is a union of
O(d)-orbits. Any O-invariant metric (linear-probe R², kNN R², R² after an orthogonal fit) is constant on
an orbit; a non-invariant metric such as MCC is not determined by the objective at all. Hence orthogonal-fit
R² is the strongest identifiability these arms define, and when a matched-prior arm's minimiser lies in the
orbit of a Gaussian-arm minimiser (e.g. whitened linear ICA, d = k: both reach orthogonal-fit R² = 1), that
metric cannot separate them; this is a statement about the metric, not a proof that a matched prior can
never change orthogonal-fit R². Check (`theory_t4.json`): rotation changes the Gaussian-target estimators by
< 4 SE (KSD exactly up to rounding) and the uniform-prior KSD by 15.8 SE.

**(ii) (Comon 1994, Theorem 11).** If s has independent, non-degenerate components, at most one Gaussian,
and z = Ms with M orthogonal has independent components, then M is a scaled permutation; unit variances
make it a signed permutation. Applied here: if the matched-prior regulariser vanishes only at its product
law and is driven to zero, z has independent components and MCC = 1. Idealisations: (1) the property
"vanishes only at the prior" is assumed — Gorham & Mackey's conditions are not verified for the
surrogate scores used (generalised normal p = 8; tanh-smoothed Laplace); (2) study 1's content is
discrete, so no continuous prior is matched exactly and the minimum is not zero. Check: d = k = 3,
independent sources. Laplace sources + Laplace prior: MCC = 1.00 in 2/3 seeds; uniform sources +
uniform-surrogate prior: gradient descent stalled in all 3 seeds at a rotated solution (MCC 0.81–0.86) whose
KSD (≈ 0.019) is 4× that of the signed permutation (≈ 0.0045) — the population claim holds, training
misses it. Gaussian target: MCC 0.74–0.91 (arbitrary rotation), KSD of the fit equal to that of the
identity, orthogonal-fit R² = 1.00 for every arm.

**Requirements on the content code.** Study 1 codes content as (cos h, sin h, scale, simplex₃(shape)).
Only the three groups are independent; within the hue pair the law (10 equally spaced hues) is invariant
under rotations by multiples of 36° and close to rotation-invariant, so no prior can pin the in-plane
angle (check, circle code: pair MCC 0.70–0.87 across seeds under the matched prior); the simplex code
has a discrete symmetry. **The cell where T4's assumptions hold best is the scale factor** (a scalar
factor independent of the others, non-Gaussian under the uniform and Laplace laws): a matched-prior effect,
if any, should appear in the per-factor MCC of scale.
