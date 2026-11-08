# GATE 0 — Related work and novelty

Status: **complete** (2026-11-07). Spend: about $5.2 of the $12 phase budget.

## Exit criteria
- [x] Every listed paper is verified and has `notes/<citekey>.md` with claim · method · overlap · open gap · BibTeX (32/32, plus 3 found by the novelty search).
- [x] `notes/related_work.md` has the overlap table with the required columns, a "covered" list, and novelty claimed only for uncovered cells.
- [x] `paper/refs.bib` contains only verified entries: 35 entries, keys = note citekeys, braces balanced, no duplicate keys.
- [x] Unverifiable items are marked. No note is UNVERIFIED; details that came from reviewer memory are listed under "Caveats" in related_work.md.

## Done
- 35 lit-reviewer calls, one paper per call, at most 4 concurrent.
- Notes (citekeys): balestriero2025lejepa, zimmermann2026kerjepa, klindt2026when,
  bardes2022vicreg, gong2021sliced, bardes2024revisiting (V-JEPA), assran2025v (V-JEPA 2),
  assran2023self (I-JEPA), vonkugelgen2021self, liu2016a, chwialkowski2016kernel,
  gorham2017measuring, anastasiou2023steins, reinert2009multivariate, xu2022kernelised (NP-KSD),
  mijoule2023stein, xu2021generalised (GKSD → AISTATS 2022 Sf-KSD), ley2017stein,
  kingma2014auto, higgins2017betavae, tolstikhin2018wasserstein, zhao2019infovae, pu2017vae,
  burgess2018shapes, locatello2019challenging, krizhevsky2009learning,
  srivastava2015unsupervised, nicollier2026expanding (SPHERE-JEPA), wu2026visreg,
  garrido2023rankme, wang2020understanding, federici2020learning.
- Added by the novelty-guard search, all reviewed: nicollier2026beyond (arXiv:2609.21656,
  Beyond Gaussian Worlds), kuang2026rectified (arXiv:2602.01456, Rectified LpJEPA),
  arnez2026sigreg (arXiv:2607.13612, SIGReg as variational free energy).
- My own checks:
  - PMLR title and pages confirmed by script for 8 entries (liub16, chwialkowski16, gorham17a,
    locatello19a, srivastava15, wang20k, garrido23a, xu22b).
  - Federici's ICLR 2020 venue confirmed from the ML Anthology record.
  - Full-text keyword scans of SPHERE-JEPA and VISReg added to their notes, because their
    reviewers had read only the abstracts.
- `notes/_work/build_bib.py` regenerates `paper/refs.bib` from the notes.

## Key findings for later phases
- **KerJEPA already uses sliced and full KSD/MMD with IMQ/Gaussian kernels as JEPA
  regularisers, with Gaussian and Laplace priors.** Our arms are therefore not new methods.
- **Beyond Gaussian Worlds (nicollier2026beyond) already shows that targets matched to the
  latent geometry (MMD on sphere/torus) improve linear identifiability.** H3 must be framed as
  a test of whether that carries over to KSD, Euclidean uniform/Laplace laws, and 3DShapes.
- **Arnez et al. state in theory only that SIGReg with fixed isotropic noise in z is an exact
  IB.** H2's σ sweep is the first empirical test among the reviewed works.
- Uncovered cells where we claim novelty:
  - JEPA-regulariser nuisance leakage under controlled, ground-truth nuisance (α);
  - d (below/above k) × σ² bottleneck sweep in JEPA;
  - matched-prior KSD for Euclidean non-Gaussian content laws on real rendered data;
  - temporal JEPA comparing EMA, EMA + sliced KSD and sliced KSD on leakage;
  - the rank-k KSD oracle as a diagnostic;
  - a JEPA-vs-autoencoder leakage comparison (low confidence).

## Decisions
- Notes were capped at about 250–400 words plus BibTeX, with an extra "Table row" section, to
  keep synthesis cheap.
- The list's papers were supplemented with 3 found by a novelty-guard search, because two of
  them (nicollier2026beyond, arnez2026sigreg) bear directly on H2 and H3. Only 2 web searches
  were run, so coverage of 2026 preprints is not exhaustive. P3 should say "among the works
  we reviewed".
- Citekeys follow the agent rule (author + year + first title word), even when the result is
  odd (assran2025v, liu2016a). The BibTeX entry keys were renamed to match.
- Years and fields are kept as fetched:
  - KerJEPA: year 2026 per arXiv's BibTeX, although id 2512 implies a Dec 2025 submission.
  - Pu: the first author is "Yunchen" per arXiv.
  - CIFAR-10: no institution field.
- The VAE-tier novelty claim is restricted to the leakage comparison, not the KSD autoencoder
  itself.

## Problems
- OpenReview, DBLP and the arXiv export API were often blocked or rate-limited, so reviewers
  fell back on arXiv BibTeX, Crossref, PMLR, ML Anthology, CVF, and NeurIPS pages.
- Some method/data details came from reviewer memory and are marked in the notes and under
  "Caveats" in related_work.md. P3 must recheck them before citing those specifics.
- I briefly wrote 3 temporary text files to /tmp for the keyword scans and deleted them
  straight after. Noted here because it is outside the repo.

## Next (P1)
- Build `stein/` per PROMPT.md P1.
- Use the IMQ kernel for KSD (gorham2017measuring).
- Use the official `lejepa` package for SIGReg.
- The matched-prior KSD needs smooth score surrogates for uniform (e.g. a smoothed box /
  generalised-normal with large shape) and Laplace (smoothed |x|). Record the choice.
