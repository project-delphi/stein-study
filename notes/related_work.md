# Related work and novelty (P0 synthesis)

Sources: the 35 per-paper notes in `notes/<citekey>.md`, each written by a `lit-reviewer`
call and listed in `gates/GATE_0.md`. All 35 are VERIFIED (title, authors, year, and venue
where one exists). Three were not on the study's list: a novelty-guard web search on
2026-11-07 found them (`nicollier2026beyond`, `kuang2026rectified`, `arnez2026sigreg`).
Every cell below paraphrases the note's "Table row". "—" means the paper does not address
the column. "(abs.)" means the reviewer read only the abstract for that cell.

## Overlap table

| Paper | Regulariser | Sliced/full | Prior / target | Nuisance | Dim. bottleneck | Info. bottleneck | Identifiability | Objective | Data |
|---|---|---|---|---|---|---|---|---|---|
| **JEPA / SSL distributional regularisers** |||||||||
| LeJEPA `balestriero2025lejepa` | SIGReg (Epps–Pulley on 1-D projections) | sliced | isotropic Gaussian | — (augmentations only) | — | — | informal (optimal-embedding argument) | JEPA | natural |
| KerJEPA `zimmermann2026kerjepa` | MMD and KSD, Gaussian/IMQ kernels; SIGReg recovered as sliced MMD | both (unsliced, finite-sliced, analytic) | Gaussian, Laplace (Student-t in theory only) | — | — (slicing dim. only) | — | — (characteristic kernel only) | JEPA (EMA student–teacher) | natural |
| Expanding SPHERE-JEPA `nicollier2026expanding` | full MMD, KSD, KDE-KL on the sphere; SIGReg/SUSReg baselines | full (+ sliced baselines) | uniform on hypersphere | — (full-text keyword check) | — | — | — | JEPA | natural (ImageNet-100, Galaxy10, textures) |
| Beyond Gaussian Worlds `nicollier2026beyond` | heat-kernel MMD matched to latent geometry; SIGReg, Matérn-MMD baselines | full | Gaussian, uniform on sphere / torus / Clifford torus, matched vs mismatched | — | partial (d vs d+2 ambient rank) | — | linear up to orthogonal/linear map (theory + affine-probe R²) | JEPA with diffusion (temporal-like) pairs | rendered (procedural robot) + simulated trajectories |
| Rectified LpJEPA `kuang2026rectified` | sorted-projection (sliced-Wasserstein-type) distribution matching | sliced | rectified generalised Gaussian, fixed (not data-matched) | — | — (appendix ablation unread) | — | — | JEPA | natural |
| VISReg `wu2026visreg` | variance + sliced-Wasserstein sketching + invariance | sliced | isotropic Gaussian | — (keyword check) | — | — | — | JEPA | natural |
| VICReg `bardes2022vicreg` | variance hinge + covariance decorrelation + invariance | full (moments only) | none (moment constraints) | implicit (augmentation invariance) | — | — | — | joint embedding | natural |
| Wang & Isola `wang2020understanding` | alignment + Gaussian-potential uniformity | full (pairwise) | uniform on hypersphere | implicit | — | — | — | contrastive | natural |
| **JEPA with EMA / stop-gradient** |||||||||
| I-JEPA `assran2023self` | EMA target + stop-grad | — | — | — (masking) | — | — | — | JEPA | natural |
| V-JEPA `bardes2024revisiting` | EMA target + stop-grad | — | — | — (masking) | implicit (narrow predictor) | — | — | temporal JEPA | natural video |
| V-JEPA 2 `assran2025v` | EMA target + stop-grad | — | — | — | — | — | — | JEPA; temporal action-conditioned stage | natural video |
| **Identifiability and views** |||||||||
| Klindt et al. `klindt2026when` | SIGReg (VICReg, InfoNCE compared) | sliced | isotropic Gaussian; non-Gaussian *data* laws tested | — (views differ by OU noise only) | — (assumes output dim = latent dim; flagged open) | — | linear up to orthogonal Q; Gaussian is the unique such law within their world class | JEPA, temporal/stationary pairs | synthetic + rendered (DMC Reacher) |
| von Kügelgen et al. `vonkugelgen2021self` | alignment + entropy / InfoNCE | — | — | **yes**: style resampled across views, discarded by invariance | yes (encoder dim = content dim) | — | block identifiability of content | contrastive / multi-view | synthetic + rendered (Causal3DIdent) |
| Federici et al. `federici2020learning` | symmetrised KL between view posteriors (MIB) | — | none (other view's posterior) | **yes**: view-specific information penalised | — | yes (β-weighted) | — (sufficiency/minimality) | multi-view IB | natural + augmented MNIST |
| Arnez et al. `arnez2026sigreg` | SIGReg (VICReg, LogDet, PairDist in theory) | sliced | isotropic Gaussian | — | — | yes, theory only: fixed isotropic noise in z under SIGReg gives an exact IB | assumed from Klindt et al. | temporal action-conditioned JEPA | none (no experiments) |
| Locatello et al. `locatello2019challenging` | KL / TC / moment penalties (β-VAE family) | — | factorised Gaussian | — | — | implicit (β) | impossible without inductive bias / supervision | autoencoder | synthetic + rendered (incl. Shapes3D) |
| RankMe `garrido2023rankme` | — (effective-rank metric) | — | — | — | rank collapse diagnosed, d not varied | — | — | evaluation of JE-SSL | natural |
| **Autoencoders** |||||||||
| VAE `kingma2014auto` | per-sample KL in ELBO | full (closed form) | N(0, I) | — | fixed latent dim | implicit | — | autoencoder | natural (MNIST, Frey Faces; from reviewer memory) |
| β-VAE `higgins2017betavae` | β-weighted KL | full | N(0, I) | — | — | yes (β > 1) | empirical disentanglement metric | autoencoder | synthetic + rendered + natural |
| WAE `tolstikhin2018wasserstein` | MMD (IMQ) or GAN on aggregate posterior | full | N(0, σ_z² I) | — | yes (d_z vs intrinsic dim discussed) | — | — | autoencoder | natural (MNIST, CelebA) |
| InfoVAE `zhao2019infovae` | MMD (main), SVGD-style Stein variant, adversarial, on aggregate posterior | full | N(0, I) | — | — | yes (MI weight α) | — | autoencoder | synthetic + MNIST (+ CIFAR qualitative) |
| Stein-VAE `pu2017vae` | SVGD transport of posterior particles (no explicit prior penalty) | full (RBF) | simple Gaussian prior, not varied | — | — | implicit | — | autoencoder | synthetic + natural |
| **Stein background (theory / testing)** |||||||||
| KSD test `liu2016a` | KSD (U-statistic) as test statistic | full | any target with known score | — | — | — | — | GOF testing | synthetic |
| Kernel GOF test `chwialkowski2016kernel` | Stein-kernel test, wild bootstrap | full | any target with known score | — | — | — | — | GOF testing | synthetic |
| Gorham & Mackey `gorham2017measuring` | KSD as sample-quality measure; IMQ kernel controls convergence where Gaussian/Matérn can fail for d ≥ 3 | full | any target with known score | — | — | — | — | MCMC diagnostics | synthetic |
| Sliced KSD `gong2021sliced` | max-sliced KSD; S-SVGD | sliced (optimised directions) | model / null density | — | — | — | — | GOF, ICA, amortised VAE posterior | synthetic + MNIST, UCI |
| NP-KSD `xu2022kernelised` | NP-KSD test (coordinate-wise conditional scores) | full | implicit generator law | — | — | — | — | GOF for generative models | synthetic + MNIST, CIFAR-10 |
| Sf-KSD / GKSD `xu2021generalised` | KSD with general Stein operators (abs.) | — | null density | — | — | — | — | GOF testing | — |
| Anastasiou et al. `anastasiou2023steins` | survey of Stein discrepancies (abs.) | — | generic | — | — | — | — | review | — |
| Reinert & Röllin `reinert2009multivariate` | exchangeable-pair bounds | — | possibly singular (rank-deficient) normal | — | — | — | — | probability theory | — |
| Mijoule et al. `mijoule2023stein` | multivariate density/score Stein operators, Stein kernels | — | any continuous density on R^d | — | — | — | — | probability theory | — |
| Ley et al. `ley2017stein` | univariate density/score Stein operators | — (1-D) | general univariate targets | — | — | — | — | probability theory | — |
| **Data** |||||||||
| 3DShapes `burgess2018shapes` | — | — | — | labelled factors (floor/wall hue, orientation usable as nuisance) | — | — | ground truth enables testing | dataset | rendered |
| CIFAR-10 `krizhevsky2009learning` | — | — | — | — (augmentation params logged by us) | — | — | — | dataset | natural |
| Moving MNIST `srivastava2015unsupervised` | — | — | — | — (motion modelled, not removed) | fixed LSTM state | — | — | autoencoder + future prediction (pixels) | synthetic (+ natural video) |
| **This study** | SIGReg, VICReg, sliced/full KSD (IMQ), MMD, EMA, oracle rank-k KSD | both | Gaussian; matched uniform/Laplace (KSD); rank-k Gaussian (oracle) | **ground-truth nuisance, independent across views, strength α; leakage R²** | **d swept below/above k** | **injected noise σ², swept against d** | linear (orthogonal fit to content factors) | JEPA, temporal JEPA, (autoencoder) | rendered (3DShapes) + natural (CIFAR-10); temporal tier on 3DShapes walks or Moving-MNIST built from MNIST (chosen in P1) |

## What the table shows is already covered (we claim no novelty here)
- **KSD and MMD as JEPA regularisers, sliced and full, with IMQ/Gaussian kernels:** KerJEPA.
  Full KSD on the sphere: SPHERE-JEPA. Our KSD/MMD arms are re-implementations of known
  regularisers, not new methods.
- **Non-Gaussian targets:** Laplace (KerJEPA), rectified generalised Gaussian (LpJEPA),
  uniform on the sphere (SPHERE-JEPA, Wang & Isola). **Targets matched to the latent law, for
  identifiability:** Beyond Gaussian Worlds, with MMD on compact manifolds (sphere, torus) and
  rendered images. Matching the target to the latent law is therefore not a new idea.
- **Linear identifiability of JEPA latents:** Klindt et al. (Gaussian, SIGReg) and Beyond
  Gaussian Worlds (manifolds). **Content identifiability from paired views with resampled
  style:** von Kügelgen et al. (contrastive). Without supervision or inductive bias,
  identification is impossible (Locatello et al.).
- **Information bottlenecks** in autoencoders (β-VAE, InfoVAE) and multi-view IB (Federici).
  "SIGReg + fixed Gaussian noise in z = an exact IB" was stated in theory (Arnez et al.) but
  not tested empirically.
- **EMA + stop-gradient JEPAs on video:** V-JEPA and V-JEPA 2. **SIGReg with temporal pairs:**
  Klindt et al. and Arnez et al.
- **MMD toward the prior on the aggregate posterior, with an IMQ kernel:** WAE-MMD.
  **Stein-flavoured autoencoders:** InfoVAE has an SVGD-style Stein variant and Pu et al.
  use SVGD.

## Uncovered cells, and the only novelty we claim
The study is an analysis study, and all claims are relative to the 35 reviewed works.
1. **Nuisance column × JEPA distributional regularisers.** No reviewed JEPA paper with a
   distributional regulariser (LeJEPA, KerJEPA, SPHERE-JEPA, Beyond Gaussian, LpJEPA,
   VISReg, Klindt) or with EMA (I-/V-JEPA) has ground-truth nuisance that is independent
   across views at a controlled strength α, and none measures nuisance leakage. The only
   rows that treat nuisance are contrastive (von Kügelgen) or multi-view IB (Federici).
2. **Dimensional × information bottleneck in JEPA.** No reviewed paper varies the embedding
   dimension d below and above the content dimension k *and* separately varies injected
   noise σ² for JEPA regularisers. Partial coverage: von Kügelgen fixes d = content dim,
   Beyond Gaussian compares d against d+2, Klindt flags d ≠ latent dim as open, and Arnez
   et al. give the noise-IB link in theory only. **H1 and H2 sit in this gap.**
3. **KSD with a prior matched to a known non-Gaussian *Euclidean* content law, on real images
   with known factors.** Beyond Gaussian Worlds covers matched *MMD* on *compact manifolds*.
   KerJEPA uses Laplace as a swap that is not matched to the data. Klindt varies the data law
   under a Gaussian target. Matched-prior *KSD* (score-only) for uniform/Laplace content laws,
   compared with Gaussian-target regularisers on identifiability, on 3DShapes: uncovered.
   **H3 tests this.** Beyond Gaussian Worlds supports the idea for MMD, so we should present
   H3 as a test of whether that finding carries over to Euclidean laws, KSD and real
   rendered data, not as a new idea.
4. **Temporal JEPA under moving nuisance: EMA vs EMA + sliced KSD vs sliced KSD alone,
   measured by leakage.** V-JEPA/V-JEPA 2 use EMA only, Klindt and Arnez use SIGReg, and no
   KSD temporal arm or nuisance-leakage measurement appears. **H5 sits here.**
5. **The rank-k Gaussian KSD "oracle"** (a singular-normal target, with theory in Reinert &
   Röllin) as a JEPA diagnostic: not in any reviewed row. We present it only as a
   diagnostic.
6. **Data column:** no reviewed JEPA-regulariser paper uses 3DShapes. Locatello et al. use
   it for VAEs. Rendered known-factor data appear in Klindt (DMC Reacher), Beyond Gaussian
   (procedural robot) and von Kügelgen (Causal3DIdent), so "rendered data" alone is not
   novel; the content/nuisance split on 3DShapes for JEPA is.
7. **VAE tier (H6), low confidence:** among the reviewed works no autoencoder pushes the
   aggregate posterior to a non-Gaussian prior with a KSD penalty, and none compares
   autoencoder leakage with JEPA leakage. InfoVAE's Stein variant was flagged by its reviewer
   for recheck, and the autoencoder literature is larger than our sample. **Do not claim
   novelty for the KSD autoencoder itself**, only for the leakage comparison.

## Caveats for later phases (P3 must respect these)
- **Facts from reviewer memory.** Some details were not read from the source, and each note
  marks them. They must not be cited in the paper without a recheck:
  - liu2016a: RBF kernel, median heuristic, bootstrap;
  - kingma2014auto: datasets;
  - garrido2023rankme: the effective-rank formula (take it from the paper or define it ourselves);
  - higgins2017betavae: method and data beyond the abstract;
  - reinert2009multivariate: method description beyond the abstract;
  - srivastava2015unsupervised: UCF-101/HMDB-51;
  - wang2020understanding: dataset list;
  - pu2017vae: the prior.
- **Abstract-only reads:** anastasiou2023steins and xu2021generalised (body unread), and
  kuang2026rectified and arnez2026sigreg (partly read).
- **Key and title quirks:**
  - zimmermann2026kerjepa: the arXiv id 2512.19605 implies a Dec 2025 first submission, but
    arXiv's BibTeX export gives year 2026; we keep the fetched value.
  - xu2021generalised: the citekey reflects arXiv v1 (2021, "GKSD"), but the entry is the
    AISTATS 2022 version, retitled "Standardisation-function KSD".
  - pu2017vae: the proceedings spell the first author "Yuchen", arXiv says "Yunchen"; the
    BibTeX uses arXiv's spelling.
  - burgess2018shapes: the README BibTeX points to the old repository URL.
  - krizhevsky2009learning: no institution field, because the official page doesn't state one.
