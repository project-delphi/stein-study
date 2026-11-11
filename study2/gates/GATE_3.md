# GATE 3 — Paper (study 2) — FINAL

## Done
- `paper2/paper.pdf`: **11 pages** (US letter, single column, 10 pt, with references and the proof appendix),
  rendered from `paper2/paper.qmd`; no warnings. Contents: abstract; introduction with study 1's three
  findings (cites `paper/paper.pdf` as `kalia2026dimension`) and a list of contributions; model and theory
  (Lemma 1, Theorem 1 + Corollary 1, Heuristic 1, Theorem 2 + Table 1, VICReg heuristic, Heuristic 2 (T3),
  Proposition 3 + Comon), each with its status; theory meets code; predictions and design; verdict table
  for P1–P18 (Table 2); results organised by prediction (E1/E2 thresholds, frontiers, E3, E4, E5); what the
  theory does not explain; related theory (from `notes/related_theory.md`, novelty claimed only as
  derivations/tests, T4 = Comon, Zimmermann et al. the closest analogue); limitations and reproducibility
  (AI agent ran experiments and drafted text and proofs; proofs checked numerically, not formally);
  Appendix A with complete proofs of Lemma 1, Theorem 1, Corollary 1, Theorem 2 (incl. a monotonicity
  argument that makes the threshold well defined) and Proposition 3, plus the numerical checks.
- Figures (vector, `paper2/figures/*.pdf|svg`, all drawn by `stein/report2.py`):
  1. `fig1_theory`: linear model, per-direction nuisance variance vs λ/λ* (Thm 1 curve, population optimum
     per regulariser, real training loop) and probe R² step vs variance share.
  2. `fig2_paths`: λ paths of leakage, nuisance share and content R² for the 6 arms with theory λ* and λ̂* CI.
  3. `fig3_scaling`: α ratio vs α(2−α) / α / 1; measured vs predicted λ̂* (log-log, ×3 band).
  4. `fig4_frontier`: content vs linear and kNN leakage along the λ path.
  5. `fig5_e3`: 2000 vs 8000 steps (probes; variance decomposition).
  6. `fig6_e4`: leakage vs σ, batch vs fixed noise.
  Tables: `tab_theory.md` (linear-model λ*), `tab_verdicts.md`, `tab_thresholds.md`, `tab_e5.md`.
- Every number in the paper is a `{{< meta n.KEY >}}` lookup into `paper2/_generated/numbers.yml`
  (406 keys) written by `stein/report2.py`; a check found no missing key.
- `paper2/refs.bib` (18 entries) built by `paper2/build_bib.py` from verified notes (17) + one manual,
  labelled entry for the companion study.
- `REPRODUCE2.md`: theory checks (commands, runtimes), the full experiment job, re-running Figure 6 from
  scratch (`stein.sweep s2_fig_e4` into an empty `STEIN_RUNS`, then `stein.repro_check2`), the paper, tests.
  Spot check in P3: 2 re-run runs matched the stored metrics exactly (max |diff| = 0 over 47 metrics).
- Code: `stein/report2.py`, `stein/repro_check2.py` (new); `stein/grids.py` gained `s2_fig_e4()` (new
  function only; old grids untouched; all 100 of its configs are already cached). 40 tests pass.

## Render commands
```
.venv/bin/python -W ignore -m stein.report2
.venv/bin/python paper2/build_bib.py
quarto render paper2/paper.qmd
```

## Decisions
- D1. Study-1 motivating numbers are recomputed from `results/lam_sens_summary.parquet` (KSD, d = 12:
  leakage 0.002 at λ = 0.2, 0.382 at λ = 1, 0.533 at λ = 5). The prompt's "0.003 / 0.578" do not match
  study 1's λ-sensitivity summary; the paper uses the generated values.
- D2. SIGReg λ̂* CI is [0.0262, 0.0327] (thresholds.parquet and verdicts.json agree); GATE_2's "0.0321"
  was a transcription slip.
- D3. Verdict words in `tab_verdicts.md` are GATE_2's verdicts (judgements), encoded in report2.py; every
  statistic beside them comes from results2/verdicts.json.
- D4. Labels follow the prompt's rule: theory.md's "Proposition 1" (empty-direction traps) is not proved,
  so the paper calls it **Heuristic 1**; T3 is **Heuristic 2**; T4(ii) is cited as Comon's theorem, not
  claimed. Only Lemma 1, Theorem 1, Corollary 1, Theorem 2 (with numerical constants) and Proposition 3
  carry those names, and each has a proof in Appendix A. Theorem 2's proof adds a short monotone
  comparative-statics step (nuisance variance non-increasing in t) that theory.md left implicit.
- D5. Quarto's typst output places the bibliography at the end of the document, so the references follow
  the appendix (said in the text). Citeproc with an IEEE CSL would need a file not in the repo.
- D6. `paper2/` is rendered on its own and not added to `_quarto.yml`, so study 1's two documents and the
  project render list stay unchanged.
- D7. The companion study has no lit note (it is this repo's study 1), so `build_bib.py` adds one manual
  `@misc` entry labelled "Companion study (study 1), unpublished report, paper/paper.pdf".
- D8. "One experiment figure" in REPRODUCE2 is Figure 6 (E4): it needs 100 runs (~45 min), the cheapest
  figure built from real images. Figure 1 (theory) reproduces from check_t2 + validate2.
- D9. In Figure 2, MMD's lowest-λ mean leakage (−7.74, probe failure on collapsed seeds) is clipped at
  −0.05 and in Figure 4 that point is omitted. Both captions say so.

## Problems
- In the synthetic real loop the probe reads leakage 0.85–0.95 *below* the threshold from residual
  nuisance variance ≤ 4×10⁻⁴ (scale invariance). The paper states this in Figure 1's caption. It is the
  same mechanism that censors cov's and VICReg's thresholds in the network.
- zimmermann2021contrastive's BibTeX (from its note) has no venue field; it renders as arXiv only.

## Next (P4)
- Independent proof referee and empirical reviewer as specified; fix findings; FINAL2.md.
