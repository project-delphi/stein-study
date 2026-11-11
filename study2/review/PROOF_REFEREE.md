# Proof referee report (P4, independent sub-agent)

Saved verbatim in substance by the P4 session: the referee could not write to /tmp, so it returned the report
as its final message. Its own scripts and outputs (independent of `theory/check_*.py`) are in
`study2/review/proof_referee/`. Line numbers refer to `paper2/paper.qmd` and `theory/theory.md` as of commit
288add8 (before P4 fixes). Resolution of each finding: `study2/gates/GATE_4.md`.

## Summary
No critical errors. Lemma 1, Theorem 1, the Theorem 2 monotonicity, Cor. 1(a,c,d for cov) and Prop. 3(i)
hold; no counterexample found. Issues: one false sentence inside a proof, unstated assumptions, and several
places where the text claims more than the theory supports. Not done: an independent MMD t* check
(`check_mmd_tstar.py` was never run), so Theorem 2's per-regulariser constants were not checked independently.

## Findings
- **F1 (major).** Cor. 1 proof (paper l.468–469; theory.md l.77): "At α = 0 … any split with P+Q = I is
  optimal" is false when d > k+m (rank(P+Q) ≤ k+m), which includes the base setting d = 12 > 11. Minimum is
  λ(d−k−m) > 0. GD at d=12,k=6,m=5,λ=1 gives min J = 1.0000. Fix: minimisers have min(d,k+m) unit
  eigenvalues and the rest 0; P+Q = I only when d ≤ k+m.
- **F2 (major).** Theorem 2 (paper l.144–149; theory.md l.98–100) needs z Gaussian for the kernel arms; the
  model only assumes zero mean, identity covariance ("which it is in this model" is false). Fix: state
  Gaussian s, n explicitly for kernel arms and Table 1.
- **F3 (major).** Cor. 1(d) is proved for covariance-dependent R only, but is invoked for KSD/SIGReg (E3;
  paper l.340–341, abstract l.22–23, l.386–387). For KSD a whitened (s, g(s)) on 320 atoms has batch KSD
  0.112 vs 1.4e-4 for Gaussian samples: not zero-cost. Fix: "consistent with the mechanism of Cor. 1(d),
  proved for cov; not shown to be cheaper for KSD/SIGReg".
- **F4 (major).** Paper l.274–275: the α = 0.5 ratios 1.27/1.20 are said to be "allowed by the trap
  mechanism's upper end", but the trap escape c_p/t_loc is α-independent and equals the global threshold at
  α = 1, so it bounds the ratio at ≈ 1. Fix: point estimates lie above the trap bound of 1, CIs include 1.
- **F5 (major).** Abstract l.20, discussion l.382–383 "they scale as α(2−α)": at α = 0.5 SIGReg and MMD CIs
  exclude 0.75; at α = 0.25 sliced KSD CI [0.50, 0.86] excludes 0.44. Fix: "lie between α(2−α) and about 1
  and reject λ* ∝ α".
- **F6 (major).** Paper l.194 "A matched product prior can therefore show its effect only on MCC", abstract
  l.25–26 "orthogonal-fit R² is unchanged, as the theory requires": Prop. 3 says only that minimisers in one
  orbit are indistinguishable by orth-fit R² (theory.md l.183–184 says so). E5 Laplace cells contradict both
  (matched − Gaussian orth-fit R² −0.50, −0.44; MCC −0.21, −0.24). Fix: qualify to the same-orbit case; the
  abstract should say uniform shows an MCC gain and Laplace lowers both; "requires" → "allows".
- **F7 (major).** Heuristic 1 (paper l.137–142; theory.md l.86–91): the ΔJ formula assumes the added
  variance is a fresh signal uncorrelated with the embedding; duplicating encoded content gives slope 0, so
  content never escapes a trap in the linear model. c(u) can exceed 1, so c_p/2 is not a hard upper end; c_p/2
  is cov-only (other arms c_p/t_loc). Fix: add the "independent signal" condition, soften "anywhere
  between", say "for cov".
- **F8 (minor–major).** Prop. 3 (l.185–191, l.483–488): batch-scaled predictor noise (σ²·diag Σ) breaks O(d)
  invariance for σ > 0 (optimal predictor cost 3.27 → 2.52 under rotation). E5 uses σ = 0, so unaffected.
  SIGReg and sliced KSD are invariant only in expectation over directions. Fix: add both conditions.
- **F9 (minor).** Theorem 1(a) proof: "tr P = k" needs d > k; for d ≤ k argue J = 0 ⇒ Q = 0, P = I_d.
  theory.md's min-J term λ(d−k−r) is negative for d < k; use (d−k−r)₊.
- **F10 (minor, theory.md only).** Cost 1−(1−α)^{2j} for j-factor features holds only for pure interactions;
  an additive two-factor feature costs 1−(1−α)² (0.75 vs 0.9375 at α = 0.5).
- **F11 (minor).** theory.md l.138: "Theorem 2 applies, so a threshold exists" for VICReg — the theorem does not
  prove existence. Existence of minimisers needs R lower semicontinuous and coercive, unstated.
- **F12 (minor).** Number mismatches between files: Cor. 1(d) J 9e-16 (theory.md) vs 9×10⁻⁷ (paper; one case
  vs max); Gaussian MCC 0.74–0.91 vs 0.76–0.91; t* "within 5%" vs "within 2%"; cov t* 1.952 vs exact
  2 − 2kc/(N−1), c = 1/(1+(k+1)/(N−1)) = 1.9542 (λ* = 0.0427 unaffected).
- **F13 (minor).** Contribution (iii) l.85 calls the matched-prior analysis "heuristic" though Prop. 3 is
  proved; l.413 von Kügelgen et al. is "the d = k case", not "d ≤ k"; l.129 "starts at 0" → "is 0 at λ* and
  rises continuously".

## Checked and found correct
Lemma 1 (2000 random pairs, 0 violations; equality iff PΣ⁺Q = 0). Theorem 1 by joint L-BFGS over (A,B,W)
(finite-difference-checked gradient), 120 settings × 12 starts incl. d<k, d=k, m<d−k, m>d−k, both
predictors, α ∈ {0.25,1}: none below the closed form, 111/120 matched it with the predicted structure, 9
trapped above it; 20,000 random (P,Q) never below. Cor. 1 (a), (c), (d) for cov. Theorem 2 reduction and
monotonicity (given F2). Finite-batch cov formula (MC 3.578 ± 0.007 vs 3.575); cov thresholds 0.0427,
0.0320, 0.0187, 0.0213 reproduce. Prop. 3(i) (KSD change 0.0 under rotation; MMD invariance). T3 per-unit
costs. c_p = 1/d matches `stein/train.py`. Other status labels match theory.md. Novelty claims match
`notes/related_theory.md`.
