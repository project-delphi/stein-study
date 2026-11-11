# Empirical review of study 2 (`paper2/paper.qmd`)

Reviewer: an independent empirical reviewer (Claude). Date: 2026-11-11. I reviewed the current text, including the uncommitted wording fixes.
Scratch work is in `/tmp/empirical_reviewer/rev2/`. That directory holds the recompute scripts `recompute.py`, `recompute2.py` and `cmp_all.py`, the logs, and the sandbox `regen/`.
I reused the earlier reviewer's material only after checking it: its partial `check_t3.log`, its `repro.log`, and the 57 runs already in `/tmp/s2_fresh_rev`.

**Summary.** The numbers are sound. I recomputed more than 60 paper numbers independently from `results2/*.parquet`, `results2/*.json` and `results/lam_sens_summary.parquet`, and every one matches `numbers.yml`. Three things reproduce exactly:
- the theory checks;
- the 100 Figure 6 runs, on all 12,400 numeric metric values, list metrics included;
- Figure 6 itself, regenerated from the re-run models.

Study 1 is untouched, and `theory/predictions.md` is frozen at the P0 commit.

The problems are in interpretation:
- One verdict, P7, rests on a probe-failure artefact.
- A failed theory-meets-code check at d = k is left out of the paper, although it contradicts the paper's own explanation of P5.
- Two headline claims in the abstract are broader than the data: "always early" and "leakage is a transient".

---

## Findings

### M1. Major: P7's "clear only for MMD" result and its "confirmed" verdict come from a probe failure

- **Location:** `paper2/paper.qmd:308-309`; `paper2/_generated/tab_verdicts.md` row P7; the cause is in `stein/verdicts2.py:71-82` (`rise_width`).
- **Problem:** `rise_width` measures the 10–90 % rise from the minimum of the path. MMD's E1 path includes λ = 2.5. There, one seed's probe returns R² = −38.73, so the 5-seed mean is −7.743. That value becomes the path minimum, and both the 10 % and the 90 % crossings land between λ = 2.5 and λ = 5. The result is a "probe width" of 0.25 decades that measures nothing.
  - The paper itself calls this level a probe failure and clips it in Fig. 2.
  - Fig. 4 and `knn_below` exclude it.
  - P7 does not exclude it.
- **Evidence:** I used the repo's own `base`, `mat` and `rise_width` (`from stein.verdicts2 import ...`).

  | MMD variant | probe width (decades) | share width (decades) |
  |---|---:|---:|
  | as stored | 0.254 | 0.806 |
  | λ = 2.5 dropped | **0.932** | 0.806 |
  | probe R² clipped at 0 | **0.932** | 0.806 |
  | failed seed dropped at all λ | 0.786 | 0.795 |

  Unaffected arms (probe / share): SIGReg 0.601 / 0.620, sliced KSD 0.616 / 0.738, KSD 0.703 / 0.738, cov 0.964 / 0.937, VICReg 0.822 / 0.515.

  With the artefact removed, the probe is narrower in 3 of 6 arms, not 4. The three remaining differences are small (0.02–0.12 decades) and have no CIs. The registered criterion, "share as steep or steeper for most arms", is then neither clearly met nor clearly failed.
- **Fix:**
  - Exclude probe-failure levels in `rise_width`: drop levels with mean leakage below −0.5, the filter `report2.py:233` already uses, or clip R² at 0.
  - Change P7's verdict to "inconclusive (3/6)".
  - Delete "The difference is clear only for MMD (0.25 vs 0.81 decades)".
  - Change the `tab_verdicts` statistic "probe narrower in 4/6 arms".

### M2. Major: the failed d = k theory-meets-code check is left out, and it contradicts the paper's explanation of P5

- **Location:** `paper2/paper.qmd:220-225` ("Theory meets code ... reproduces Theorem 1") and `:299-301` ("the network trades content for nuisance, which the linear model at d ≤ k never does because its content is free"). Source: `results2/validation2.json` (`A.d6`, `A.d6_nuisance_runs`), `stein/validate2.py:11` and `:95-105`, `study2/gates/GATE_1.md:97-103`.
- **Problem:** The real training loop ran the linear model itself: the `synth` linear-Gaussian views, the `linear` encoder, the linear predictor and the cov arm, at d = 6 = k.
  - It ended with nuisance in one direction and one content direction dropped (content R² ≈ 0.83, leakage ≈ 0.20).
  - This happened in 0/6 seeds at q = 2, 5/6 at q = 5 and 6/6 at q = 20.
  - At q = 5 it persisted to 32,000 steps (GATE_1).
  - The check was written as a gate, failed, and was then demoted to an "observation" (validate2 docstring: "written first as a gate; it failed at q = 5").

  The paper says nothing of this. Its explanation of the P5 failure, that the nonlinear network does what the linear model never does, conflicts with the linear model doing the same thing under the same training loop. Theorem 1(a) is about global minimisers, so the theorem is not contradicted. The explanation is.
- **Evidence:** `validation2.json` `d6_nuisance_runs = {'2.0': 0, '5.0': 5, '20.0': 6}`. At q = 5 and q = 20, the d6 rows have `nuis_share` 0.10–0.28, `leak_r2` 0.19–0.36 and `content_r2` 0.70–0.84. I reran `stein.validate2`: the output is identical apart from timing.
- **Fix:** In "Theory meets code" and in the P5 paragraph, report that at d = k the real loop on the linear model reaches a non-global state with nuisance in 5/6 and 6/6 seeds. Reword P5's explanation: the d = k leakage is reproduced by training dynamics even in the linear model, so it cannot be attributed to nonlinearity. Also mention that the gate was demoted after it failed.

### M3. Major: "always early" in the abstract and the discussion is contradicted by the linear-encoder thresholds

- **Location:** `paper2/paper.qmd:22` (abstract: "0.29–0.79 times the predicted values (always early)"), `:281` ("Every uncensored threshold is *earlier* than the linear model"), `:401`.
- **Problem:** The range and the "always" cover only the four kernel arms at the base setting.
  - The paper's own E2 rung gives thresholds *later* than the linear model: KSD 1.25 (1.33×) and cov 0.403 (9.44×).
  - Fig. 3b plots these as open markers among "every uncensored setting".
  - All the other uncensored settings (α = 0.5 and 0.25, d = 24) are indeed early. I checked each: for example, MMD at α = 0.5 is 12.75 against 12.8.
- **Evidence:** I recomputed λ̂* with the frozen estimator (my own implementation) from `runs2.parquet`. Linear-encoder cov is 0.4029 [0.368, 0.460] and KSD 1.2493 [1.109, 1.601]. The theory values are 0.0427 and 0.939.
- **Fix:** Write "for the convolutional encoder, 0.29–0.79 times the predicted values (always early); with a linear pixel encoder, 1.3× (KSD) and 9.4× (cov)". At :281, write "Every uncensored conv-encoder threshold".

### M4. Major: "Leakage is also a transient ... the network replaces nuisance with nonlinear content features" (abstract) is broader than the E3 data

- **Location:** `paper2/paper.qmd:22-24`, `:340-358`.
- **Problem:** In 2 of 4 E3 cells, nuisance remains decodable at 8000 steps.
  - **SIGReg λ = 0.32:** linear leakage 0.35 → 0.17; kNN leakage 0.74 → 0.67, barely changed.
  - **KSD λ = 3.2:** linear leakage 0.48 → 0.07; kNN leakage 0.81 → 0.50.

  Only the two lower-λ cells (KSD 0.8, SIGReg 0.08) show both linear and kNN leakage collapsing (kNN 0.64 → 0.19 and 0.62 → 0.12). "Replaces nuisance" therefore holds for linear decodability in 2–3 cells. It does not hold for nuisance information.
- **Evidence:** Recomputed from `runs2.parquet` (5 seeds per cell). The 2000 → 8000 step values match the `e3_*` keys exactly.
- **Fix:** Write "linear leakage falls in all four cells, and to about 0 in two. kNN leakage stays at 0.50–0.67 at the higher λ levels, so nuisance becomes less linearly accessible rather than absent there." Soften the abstract to match.

### m1. Minor: the P1 refutation and the "cov has no level ≤ 0.10" claim sit on a knife edge, at a collapsed embedding

- **Location:** `paper.qmd:267-274`, `:333`; tab_verdicts P1.
- **Evidence:**
  - Lowest-λ mean leakage is cov 0.10167 and VICReg 0.10109, against the registered 0.10.
  - Cov per-seed values are 0.205, 0.235, 0.065, 0.001 and 0.003, with CI [0.015, 0.189].
  - At this level cov's embedding standard deviation is about 1e-4 and content R² is 0.049.
  - The verdict follows the registered criterion literally, but it rests on probe readings of a numerically collapsed embedding, and a 0.002 change would flip it.
- **Fix:** Say so in the P1 paragraph: "refuted by 0.002 (cov) and 0.001 (VICReg), CIs include 0.10, at near-collapsed embeddings".

### m2. Minor: P2 "refuted" rests on point estimates, and both decisive CIs straddle the factor-3 boundary

- **Location:** `paper.qmd:279-281`; tab_verdicts P2.
- **Evidence:** I converted the bootstrap CIs of λ̂* into ratios against the theory values:
  - KSD: [0.2716, 0.3452] / 0.939 = [0.29, 0.37].
  - Sliced KSD: [1.62, 2.78] / 7.33 = [0.22, 0.38].
  - Both include 1/3. The criterion ("outside the factor for ≥ 2 of 5 arms") does not mention CIs, so the label is defensible, and the text does say "just outside".
- **Fix:** Add "(both CIs include 1/3)" to the text and to the table statistic.

### m3. Minor: P4's "confirmed" leaves out that 2 of 4 arms have α = 0.5 point estimates outside the registered interval

- **Location:** tab_verdicts P4; abstract `:20-21` ("lie between α(2−α) and about 1"); `:399`.
- **Evidence:** P4 registered [0.75, 1] at α = 0.5. KSD is 1.27 [0.94, 1.57] and sliced KSD 1.20 [0.77, 1.65]. The registered *against* criterion only concerns α = 0.25, where all four arms (0.50–0.65) pass. The body text discusses this honestly, but "about 1" for 1.27 is generous.
- **Fix:** Label the verdict "not refuted (α = 0.25: 4/4 in range; α = 0.5: 2/4 above 1, CIs include 1)". In the abstract, write "between α(2−α) and 1.3".

### m4. Minor: three verdict labels are stronger or weaker than the registered criteria

- **P16 "confirmed"** (`:386`, tab_verdicts):
  - The prediction is "differs by < 0.02, or favours the Gaussian target". At uniform λ = 0.04 the difference is +0.026 [−0.05, 0.11], in favour of the matched prior.
  - The positive claim is therefore not met in that cell. Only the against criterion is avoided.
  - Suggested label: "not refuted".
- **P9 "inconclusive (cov)":**
  - The registered against criterion is "cov values outside both ranges".
  - Linear-rung cov gives v = 0.3006 at 2λ̂*, inside [0.3, 0.7] by 0.0006, and 0.526 at 5λ̂*, outside [0.6, 1.0].
  - Conv cov (share-based) gives 0.63, inside, and 1.44, outside.
  - Neither rung is outside both ranges, so by the registered rule the result is "not refuted (one of two ranges met at each rung)". Say that explicitly.
- **P8 "refuted (cov)":**
  - Cov's α ratio is 0.69 [0.55, 0.87], below the registered [0.75, 1].
  - KSD's α ratio is 1.06 [0.82, 1.35], above it on the point estimate.
  - The label and the table statistic mention only cov's location (9.4×).

### m5. Minor: P10 is reported only at the linear rung, where it is refuted; its conv-rung outcome is not reported

- **Location:** `paper.qmd:317-318`; tab_verdicts P10.
- **Evidence:** P10 is "Same as P3, read on the E2 slice". At the conv encoder, cov is censored below 0.020, so λ̂*_KSD / λ̂*_cov > 0.296 / 0.020 = 14.8 (> 5). The share-based ratio is 8.7 [7.1, 38.8], which `numbers.yml` already holds as `p10_share` but the paper does not use. Both are consistent with P10.
- **Fix:** Report "refuted at the linear rung (3.1 [2.6, 4.0]); consistent at the conv rung (> 14.8; share-based 8.7)".

### m6. Minor: "VICReg shows no σ effect on leakage (all CIs include 0)" hides a consistent negative sign

- **Location:** `paper.qmd:375-377`.
- **Evidence:**
  - All 8 VICReg leakage contrasts against σ = 0 are negative, from −0.020 to −0.156.
  - Under the stored unpaired bootstrap (`verdicts2.boot_diff`), some upper bounds barely include 0: λ = 5, fixed σ = 1 is −0.156 [−0.301, 0.001]; λ = 1.25, fixed σ = 0.5 is −0.085 [−0.166, 0.005].
  - Under a seed-paired bootstrap, which I computed, 5 of 8 contrasts exclude 0, all negative.
  - Either way, "no effect" undersells the result: σ, if anything, *lowers* VICReg leakage.
- **Fix:** Write "σ does not raise VICReg leakage. All 8 point estimates are negative (−0.02 to −0.16), and the unpaired CIs just include 0."

### m7. Minor: part of P14 was not tested and the paper does not say so

- **Location:** `paper.qmd:365-370`.
- **Problem:** The registered threshold ratios λ*(σ)/λ*(0) = 0.84 / 0.26 were never measured. E4 has two λ levels and no λ path under σ, so no λ̂*(σ) exists.
- **Fix:** Add one sentence saying so.

### m8. Minor: the t4 invariance sentence leaves KSD's z out of the stated bound

- **Location:** `paper.qmd:516-518`.
- **Evidence:** `theory_t4.json` invariance z-values: sliced KSD −0.70, SIGReg −0.15, MMD −0.61, **KSD −6.62**. The relative change for KSD is 1.0×10⁻⁷, consistent with floating-point error against a near-zero standard error. `report2.py:175` computes `t4_z_max` over the three other arms only. As written, "|z| ≤ 0.7 for the Gaussian-target estimators" reads as if it includes KSD.
- **Fix:** Write "|z| ≤ 0.7 for sliced KSD, SIGReg and MMD. KSD is invariant sample by sample (relative change 1×10⁻⁷; its z = −6.6 reflects floating-point error against a near-zero SE)."

### m9. Minor: BibTeX errors

- **`balestriero2022contrastive`** (`paper2/refs.bib`): `volume = {36}` is wrong. papers.nips.cc lists the paper in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. The note `notes/balestriero2022contrastive.md:36` admits the volume was inferred from "36th Conference". Fix: change it to 35 (fix the note and rerun `paper2/build_bib.py`).
- **`zimmermann2026kerjepa`:** arXiv v1 is dated 22 Dec 2025 and v2 13 Oct 2026. Year 2026 conflicts with the convention stated in the `zimmermann2021contrastive` note (year = first submission). Fix: use 2025, or cite v2 explicitly.
- **`zimmermann2021contrastive` and `simon2023stepwise`:** both are `@misc` with a note, although they appeared at ICML 2021 (PMLR 139) and ICML 2023 (PMLR 202). Cosmetic; `@inproceedings` would be better.
- **All other entries** match their sources on authors, year and venue: assran2023self, balestriero2025lejepa, bardes2022vicreg, burgess2018shapes, comon1994independent (Signal Processing 36(3):287–314, 1994; secondary sources confirm that Thm 11 is the identifiability theorem the paper cites), gong2021sliced, gorham2017measuring, jing2022understanding, liu2016a, shah2020pitfalls, shwartzziv2023information (NeurIPS 36 is correct for 2023), tian2021understanding and vonkugelgen2021self. kalia2026dimension is the companion report.

### m10. Minor: REPRODUCE2.md names the wrong training device

- **Location:** `REPRODUCE2.md:41-42` ("Training is deterministic on CPU") and the "M1 Pro, CPU" runtime header.
- **Evidence:** All 100 fresh and all 100 stored E4 `result.json` files record `device: mps`. GATE_1 also says "MPS training deterministic". Determinism is therefore shown on MPS only. CPU and CUDA reproductions were not tested.
- **Fix:** Say "deterministic on Apple MPS (tested); other devices untested".

### m11. Minor: the page budget is exhausted, and the committed PDF was stale

- **Location:** `paper2/paper.pdf`.
- **Evidence:** The committed PDF had 11 pages and predated the wording fixes; it lacked the new text. Rendering the current `.qmd` gives exactly **12 pages**, with the appendix on pp. 10–11 and the references on p. 11 onward. I re-rendered `paper2/paper.pdf` in place. Fixes M1–M4 add text and will push the paper over 12 pages unless something is cut.

### m12. Minor: some result statements are hand-typed, not generated

- **Location:** `paper.qmd`. All of the following are results typed into the text rather than `meta` lookups:
  - `:221` "between 0.8λ* and 1.25λ* in both seeds"
  - `:224` "within 0.03"
  - `:278` "all CIs exclude 1"
  - `:280` "SIGReg and MMD fall within the registered factor 3"
  - `:289` "their CIs include 1"
  - `:297` "all six arms"
  - `:308` "4 of 6 arms"
  - `:310` "two to three times later"
  - `:392` "vanishes at λ = 0.04"
- **Evidence:** I checked each one, and all are correct except "4 of 6" (see M1):
  - Validation onset: v = 4e-4 at 0.8λ*, then 0.14 and 0.17 at 1.25λ*. 9/10 runs are within 0.03.
  - P3 CIs: [8.9, 12.5], [5.3, 9.6] and [4.8, 8.8].
  - Share-to-probe crossing ratios: 2.18 (SIGReg), 2.73 (KSD), 2.75 (sliced KSD), 2.40 (MMD).
  - No literal number is a hand-typed metric value.
- **Fix:** Generate the counts and qualifiers in `report2.py` so they cannot drift (P7's did).

### Observation (no fix required)
Grids were placed to bracket each arm's linear-model threshold, and outside a grid the estimator can only report censoring. Agreement "within a factor of about 3", and to some extent the P3 ordering across thresholds that span three orders of magnitude, is therefore partly supported by the design. The grids do overlap, and the successive ratios 10.5, 7.3 and 6.3 have CIs well above 1, so P3 is still a real test. Readers should know the location test is conditional on the grid.

---

## Checked and found correct

### 1. Number provenance
130 `meta` keys are used in the `.qmd`, all present in `numbers.yml`; the rendered PDF contains no `??` and no literal `meta n.`. I recomputed these independently, with my own threshold estimator and bootstrap and my own contrasts from `runs2.parquet`, or read them directly from the json files. All match `numbers.yml` at the printed precision:

- **Thresholds (E1):**
  - λ̂* at α = 1, d = 12: SIGReg 0.0282 [0.0262, 0.0321], sliced KSD 2.15 [1.63, 2.78], KSD 0.296 [0.272, 0.345], MMD 13.6 [12.5, 16.2]. The CIs differ from the stored ones only by bootstrap noise.
  - Cov and VICReg are censored (< 0.020, < 0.31).
  - Ratios to theory: 0.72, 0.29, 0.32, 0.79.
  - P3 ratios: 10.52, 7.26, 6.30.
- **α ratios:**
  - α = 0.5: SIGReg 0.918, sliced KSD 1.20, KSD 1.27, MMD 0.941.
  - α = 0.25: 0.517, 0.646, 0.549, 0.496.
- **d = 24 ratios:** KSD 0.363, SIGReg 0.760, MMD 0.814; sliced KSD censored.
- **d = 6:**
  - λ̂*: cov 0.362, SIGReg 0.0412, sliced KSD 5.72, KSD 1.235; MMD < 20, VICReg < 2.5.
  - Maxima: 0.33–0.67.
  - Leakage at d = 6 is below d = 12 at all 24 shared levels.
- **Linear rung:**
  - Cov 0.403, ratio 9.44, α ratio 0.688.
  - KSD 1.249, ratio 1.33, α ratio 1.06.
  - P10 ratio: 3.10.
- **E3:**
  - Leakage changes −0.363, −0.412, −0.331, −0.177.
  - KSD 0.8: unexplained share 0.59 → 0.84; content R² 0.60 → 0.32; kNN content 0.85 → 1.00; kNN leakage 0.64 → 0.19.
- **E4:**
  - KSD λ = 0.1 contrasts: +0.023, +0.662, +0.155, +0.637.
  - KSD λ = 0.2 contrasts: +0.302, +0.646, +0.454, +0.671.
  - Fixed − batch: +0.132 and +0.152.
  - Embedding std (KSD λ = 0.1): 0.80 → 0.41.
  - Content share (KSD λ = 0.2): 0.44 → 0.52 → 0.51.
  - All VICReg contrasts match.
- **E5:** all 24 table entries (point estimates), for example uniform λ = 0.2: orthogonal-fit R² −0.022, MCC +0.087, hue pair +0.250, scale +0.218, leakage +0.118, content −0.063.
- **Frontier and collapse:**
  - Best content R² at leakage ≤ 0.05: KSD 0.727, MMD 0.629, SIGReg 0.573, sliced KSD 0.572.
  - Best content R² at leakage ≤ 0.20: cov 0.385, VICReg 0.258.
  - VICReg maximum content R²: 0.395.
  - Leakage at content R² = 0.5 (MMD interpolation): 0.16.
  - Lowest-λ collapse statistics: cov std 0.000, content 0.049, rank 6.19; VICReg rank 2.38.
  - kNN leakage range at linear leakage < 0.05: 0.13–0.22.
- **Study 1** (`results/lam_sens_summary.parquet`): KSD leakage 0.0016, 0.382 and 0.533 at λ = 0.2, 1 and 5.
- **Theory:**
  - Every entry of `tab_theory`: λ* values, the t*/t_quad values 0.97, 1.9, 5.3, 2.3 and 2.3, and 12 at d = 24.
  - cov t* = 1.952; v at 2λ*: 0.22, 0.40, 0.37.
  - VICReg: no nuisance up to λ = 4.6; share 0.17 at λ = 10.
  - T3: 0.94 / 0.78 / 0.25, 0.88, 0.002, 0.0424.
  - T1: traps 21/60 and 6/135; J_nonlinear 8.7e-7 against a linear minimum of 9.5.
  - T4: Laplace MCC 0.88/1.00/1.00, uniform 0.81–0.86, Gaussian 0.76–0.91; KSD fit 0.019 against 0.0045.
  - Validation: 0.467/0.469, 0.761/0.764, 0.466, 0.760, 9/10.

### 2. Pre-registration
- `git log --follow theory/predictions.md` lists only `0cf2bc5 study2: P0 complete`.
- `git diff 0cf2bc5 HEAD -- theory/predictions.md` is empty, and the working tree is clean.
- The P0 commit is dated 2026-11-10 17:39. The earliest study-2-only network run (cov or linear encoder) finished at 18:12.
- The paper reports all 18 predictions: 7 refuted (P2, P5, P10, P11, P14, P15, P17), P1 and P8 refuted in part, P12 descriptive, and P18 not tested.

### 3. Verdicts against CIs and criteria
- These follow the registered criteria as stated: P3, P5, P6, P11 (all four drops > 0.05 with CIs excluding 0), P13 (leakage part), P14, P15 (KSD; at λ = 0.2 the cell CIs fail to overlap by 0.001) and P17.
- Exceptions: M1 and m1–m5.

### 4. References
See m9. All 18 entries correspond to real works with correct authors. Venue and year are correct except where m9 says otherwise.

### 5. Render and study-1 integrity
- `quarto render` succeeds with no warnings and produces 12 pages.
- `git diff c3dd0d8 -- paper/ blog/ results/ gates/ FINAL.md REPRODUCE.md PROMPT.md notes/related_work.md` is empty, and `git status` shows nothing under those paths.

### 6. Reproduction
I ran the following. Each command stayed under 40 minutes.

- **Theory checks:**
  - check_t1, t2 and t4 had already been rerun (only `runtime_s` changed).
  - I reran `.venv/bin/python theory/check_t3.py` (1148 s, prints `FAIL`, which is expected). `results2/theory_t3.json` is identical to the committed file apart from `runtime_s`, including `pass: false`.
  - I reran `.venv/bin/python -W ignore -m stein.validate2` (PASS, 7 s); `validation2.json` is identical apart from timing.
  - `.venv/bin/python -m pytest -q tests`: 40 passed.
- **Figure 6 sweep:**
  - `STEIN_RUNS=/tmp/s2_fresh_rev .venv/bin/python -W ignore -m stein.sweep s2_fig_e4 --workers 6` resumed at 62/100 and finished 100/100 with 0 failures (937 s).
  - `.venv/bin/python -m stein.repro_check2 /tmp/s2_fresh_rev`: 100 runs, largest |diff| = 0, exit 0.
  - `repro_check2` compares scalars only, so I also ran my own `cmp_all.py`, which includes list metrics: 100 runs, 12,400 numeric values, max |diff| = 0, identical key sets.
- **Figure 6 regenerated from the re-run models:**
  - Sandbox `/tmp/empirical_reviewer/rev2/regen`, built by `make_regen.sh`: the repo's `runs/` as symlinks, with the 100 E4 hashes pointing to `/tmp/s2_fresh_rev`.
  - In the sandbox I ran `stein.analyze2`, `stein.verdicts2` and `stein.report2`.
  - Identical to the repo: `runs2`, `cells2`, `thresholds` and `frontier` (all `.parquet`), `verdicts.json`, `numbers.yml` and all four `tab_*.md`.
  - `fig6_e4.svg` (also fig1 and fig2) is identical after normalising matplotlib's random SVG ids and the date stamp.
- **Files I regenerated in the repo** (all by documented commands): `results2/theory_t3.json` and `results2/validation2.json` (timing field only), and `paper2/paper.pdf` (re-rendered from the current `.qmd`).
