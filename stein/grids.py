"""P2 grids. `python -m stein.sweep <name> --workers 6 --out results/<x>.parquet` calls the
function of that name. Each returns a list of config dicts (overrides of train.DEFAULTS).

Protocol (GATE_2 "Decisions"):
- tuning seeds {100, 101}; held-out seeds {1, 2, 3, 4, 5} (seed 0 was used by the P1 pilots);
- every tuned arm gets exactly 3 configs: lambda x {1/5, 1, 5} around the pilot centre, or
  EMA tau in {0.99, 0.996, 0.999};
- selection (stein/select_tuning.py): highest mean validation linear content R^2 over the two
  tuning seeds (CIFAR: class accuracy); runs with a non-finite loss are excluded.
"""
import json, os

from .train import resolve, cfg_hash

TUNE_SEEDS = [100, 101]
HELD_SEEDS = [1, 2, 3, 4, 5]
SEL_PATH = "results/tuning_selected.json"

BASE = dict(tier="shapes", d=12, alpha=1.0, sigma=0.0, law="uniform", prior="gaussian",
            device="mps", steps=2000)
CENTRE = dict(vicreg=5.0, sigreg=0.05, sksd=5.0, ksd=1.0, mmd=10.0, oracle=0.1)
VAE_CENTRE = dict(betavae=1.0, ksd=10.0, mmd=100.0)
TAUS = [0.99, 0.996, 0.999]
MULTS = [0.2, 1.0, 5.0]

SHAPES_ARMS = ["none", "vicreg", "sigreg", "sksd", "ksd", "mmd", "ema", "oracle"]
GAUSS_TARGET = ["sigreg", "sksd", "ksd", "mmd"]
TEMPORAL_ARMS = ["ema", "ema+sksd", "ema+sigreg", "sksd", "vicreg"]
CIFAR_ARMS = ["none", "vicreg", "sigreg", "sksd", "ksd", "mmd", "ema"]   # no oracle: k unknown
VAE_ARMS = ["betavae", "ksd", "mmd"]


def _cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    if c["tier"] == "temporal":
        c.setdefault("T", 3)
    return c


def _levels(arm, centre_table=CENTRE):
    """The 3 tuning configs of an arm, as override dicts."""
    if arm == "none":
        return [dict(lam=0.0)]
    if arm == "ema":
        return [dict(lam=0.0, ema_tau=t) for t in TAUS]
    reg = arm.split("+")[-1]
    return [dict(lam=round(centre_table[reg] * m, 6)) for m in MULTS]


def _dedup(cfgs):
    seen, out = set(), []
    for c in cfgs:
        h = cfg_hash(resolve(c))
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out


# ----------------------------------------------------------------------------- tuning slots
# A "slot" is (name, base overrides, arm); the selected level is stored per slot name.

def tuning_slots():
    s = []
    for a in SHAPES_ARMS:
        if a != "none":
            s.append((f"shapes/{a}", dict(), a, CENTRE))
    # H3: matched-prior KSD arms tuned at their own law; at the Laplace law every H3 arm is
    # re-tuned (at the Gaussian law all arms target N(0, I) and use the base selection).
    for a in ["ksd", "sksd"]:
        s.append((f"shapes/{a}@uniform-prior=uniform", dict(law="uniform", prior="uniform"), a, CENTRE))
        s.append((f"shapes/{a}@laplace-prior=laplace", dict(law="laplace", prior="laplace"), a, CENTRE))
    for a in GAUSS_TARGET + ["oracle"]:
        s.append((f"shapes/{a}@laplace", dict(law="laplace"), a, CENTRE))
    for a in TEMPORAL_ARMS:
        s.append((f"temporal/{a}", dict(tier="temporal"), a, CENTRE))
    for a in CIFAR_ARMS:
        if a != "none":
            s.append((f"cifar/{a}", dict(tier="cifar"), a, CENTRE))
    for a in VAE_ARMS:
        s.append((f"vae/{a}", dict(tier="vae"), a, VAE_CENTRE))
    # H6 carry-over of H3: KSD autoencoder with a matched (Laplace) vs Gaussian prior, Laplace law
    s.append(("vae/ksd@laplace-prior=laplace", dict(tier="vae", law="laplace", prior="laplace"), "ksd", VAE_CENTRE))
    s.append(("vae/ksd@laplace", dict(tier="vae", law="laplace"), "ksd", VAE_CENTRE))
    return s


def tune():
    cfgs = []
    for name, over, arm, ctab in tuning_slots():
        for lv in _levels(arm, ctab):
            for sd in TUNE_SEEDS:
                cfgs.append(_cfg(**over, arm=arm, seed=sd, **lv))
    return _dedup(cfgs)


def tune_tiny():
    """Smoke test of the job: 1 config per tier, 50 steps (distinct hashes, never reported)."""
    out = []
    for name, over, arm, ctab in [tuning_slots()[i] for i in (0, 3, 9, 18, 24, 30)]:
        out.append(_cfg(**over, arm=arm, seed=TUNE_SEEDS[0], steps=50, **_levels(arm, ctab)[0]))
    return out


# ----------------------------------------------------------------------------- held-out grid

def _sel():
    return json.load(open(SEL_PATH))


def _arm_cfg(sel, slot, tier_over, arm, **extra):
    """Config of `arm` at the level selected for `slot` (none: lam 0)."""
    if arm == "none":
        lv = dict(lam=0.0)
    else:
        lv = sel[slot]["level"]
    return _cfg(**tier_over, arm=arm, **lv, **extra)


def heldout_blocks():
    sel = _sel()
    blocks = []
    # H1: d below/above k (k = 6) x all arms
    h1 = [_arm_cfg(sel, f"shapes/{a}", {}, a, d=d) for d in [3, 6, 12, 24] for a in SHAPES_ARMS]
    blocks.append(("H1", h1))
    # H4: alpha at 3 levels x {EMA, distributional regularisers}
    h4 = [_arm_cfg(sel, f"shapes/{a}", {}, a, alpha=al) for al in [0.0, 0.5, 1.0]
          for a in ["ema", "vicreg"] + GAUSS_TARGET]
    blocks.append(("H4", h4))
    # H2: sigma in {0, 0.5, 1.0} x d in {12, 24}
    h2 = [_arm_cfg(sel, f"shapes/{a}", {}, a, d=d, sigma=s) for d in [12, 24] for s in [0.0, 0.5, 1.0]
          for a in ["ema", "vicreg"] + GAUSS_TARGET]
    blocks.append(("H2", h2))
    # H3: latent law x {matched-prior KSD / sliced KSD, Gaussian-target regularisers, oracle}
    h3 = []
    for a in GAUSS_TARGET + ["oracle"]:
        h3.append(_arm_cfg(sel, f"shapes/{a}", {}, a, law="uniform"))
        h3.append(_arm_cfg(sel, f"shapes/{a}", {}, a, law="gaussian"))      # ksd here = matched prior
        h3.append(_arm_cfg(sel, f"shapes/{a}@laplace", {}, a, law="laplace"))
    for a in ["ksd", "sksd"]:
        h3.append(_arm_cfg(sel, f"shapes/{a}@uniform-prior=uniform", {}, a, law="uniform", prior="uniform"))
        h3.append(_arm_cfg(sel, f"shapes/{a}@laplace-prior=laplace", {}, a, law="laplace", prior="laplace"))
    blocks.append(("H3", h3))
    # H5: temporal tier at the base setting
    blocks.append(("H5", [_arm_cfg(sel, f"temporal/{a}", dict(tier="temporal"), a) for a in TEMPORAL_ARMS]))
    # natural-image tier: one slice at the base setting
    blocks.append(("CIFAR", [_arm_cfg(sel, f"cifar/{a}", dict(tier="cifar"), a) for a in CIFAR_ARMS]))
    # H6: VAE tier at matched d = 12, plus the matched-prior carry-over at the Laplace law
    h6 = [_arm_cfg(sel, f"vae/{a}", dict(tier="vae"), a) for a in VAE_ARMS]
    h6.append(_arm_cfg(sel, "vae/ksd@laplace-prior=laplace", dict(tier="vae"), "ksd", law="laplace", prior="laplace"))
    h6.append(_arm_cfg(sel, "vae/ksd@laplace", dict(tier="vae"), "ksd", law="laplace"))
    blocks.append(("H6", h6))
    return blocks


def heldout():
    """Blocks in priority order (the cut order puts VAE last, then CIFAR); all 5 seeds of a
    config run together so a cut-off job leaves complete configs."""
    cfgs = []
    for _, blk in heldout_blocks():
        for c in blk:
            for sd in HELD_SEEDS:
                cfgs.append({**c, "seed": sd})
    return _dedup(cfgs)


def heldout_h5():
    """P3 reproduction check: the H5 (temporal) block alone, 5 held-out seeds (same configs as heldout())."""
    blk = dict(heldout_blocks())["H5"]
    return _dedup([{**c, "seed": sd} for c in blk for sd in HELD_SEEDS])


def slice_membership():
    """hash-without-seed -> list of slices the config belongs to (used by the analysis)."""
    out = {}
    for name, blk in heldout_blocks():
        for c in blk:
            key = cfg_hash(resolve({**c, "seed": -1}))
            out.setdefault(key, []).append(name)
    return out


def heldout_tiny():
    """Smoke test of handoff 2: first config of each block, 1 seed, 50 steps (never reported)."""
    return [{**blk[0], "seed": HELD_SEEDS[0], "steps": 50} for _, blk in heldout_blocks()]


def lam_sens():
    """Robustness slice (P2, foreground): every tuned level of the 3DShapes base arms on the
    held-out seeds, so the lambda-leakage trend seen in tuning is checked on held-out seeds.
    The selected level is shared with H1 (deduplicated by hash)."""
    cfgs = []
    for a in ["vicreg", "sigreg", "sksd", "ksd", "mmd", "oracle"]:
        for lv in _levels(a):
            for sd in HELD_SEEDS:
                cfgs.append(_cfg(arm=a, seed=sd, **lv))
    return _dedup(cfgs)
