"""Single-run trainer + evaluator. One entry point for all tiers:

    .venv/bin/python -m stein.train configs/<file>.yaml [key=value ...]

Runs are keyed by a hash of the resolved config; a finished run (runs/<hash>/result.json)
is skipped, so reruns are idempotent.
"""
import hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .data import Shapes, Cifar, content_code, nuisance_code, K_CONTENT, SH
from .losses import Regulariser, Prior
from .models import Encoder, Predictor, Decoder, make_ema, ema_update
from . import metrics as M

DEFAULTS = dict(
    tier="shapes",          # shapes | cifar | temporal | vae
    arm="sigreg",           # none vicreg sigreg sksd ksd mmd ema oracle | temporal: ema, ema+sksd, ema+sigreg, sksd, vicreg | vae: betavae, ksd, mmd
    d=12, sigma=0.0, alpha=1.0, law="uniform", prior="gaussian", lam=1.0,
    seed=0, steps=2000, batch=256, lr=1e-3, wd=0.0, ch=32, hidden=256,
    ema_tau=0.999, num_slices=32, sigreg_impl="lejepa",
    ema_target_norm="ln", oracle_eps=0.3, ema_loss="mse", pred_lr_mult=10.0, pred_norm="none", T=4, noise=0.02,
    n_probe_fit=10000, n_probe_eval=5000, device="cpu", threads=1,
)


ARMS = {
    "shapes": {"none", "vicreg", "sigreg", "sksd", "ksd", "mmd", "ema", "oracle"},
    "cifar": {"none", "vicreg", "sigreg", "sksd", "ksd", "mmd", "ema", "oracle"},
    "temporal": {"ema", "ema+sksd", "ema+sigreg", "sksd", "vicreg", "sigreg", "none"},
    "vae": {"betavae", "ksd", "mmd"},
}


def resolve(cfg):
    c = dict(DEFAULTS)
    unknown = set(cfg or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys {unknown}")
    c.update(cfg or {})
    if c["tier"] not in ARMS or c["arm"] not in ARMS[c["tier"]]:
        raise ValueError(f"bad tier/arm {c['tier']}/{c['arm']}")
    if c["law"] not in ("uniform", "gaussian", "laplace") or c["prior"] not in ("uniform", "gaussian", "laplace"):
        raise ValueError("bad law/prior")
    return c


def cfg_hash(c):
    keep = {k: v for k, v in c.items() if k not in ("device", "threads")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:12]


def bottleneck(z, sigma):
    """Information bottleneck: additive Gaussian noise with per-dimension std = sigma times the
    (detached) batch std of z, so the per-dimension SNR is 1/sigma^2 whatever the scale of z."""
    if sigma <= 0:
        return z
    return z + sigma * z.detach().std(0, keepdim=True) * torch.randn_like(z)


# ----------------------------------------------------------------------------- embedding helper

@torch.no_grad()
def embed(enc, x, device, bs=1000, dims=None):
    """Frozen embeddings; dims = number of leading output dims to keep (beta-VAE: mu only)."""
    enc.eval()
    out = [enc(x[i:i + bs].to(device))[:, :dims].cpu() for i in range(0, len(x), bs)]
    enc.train()
    return torch.cat(out).double().numpy()


# ----------------------------------------------------------------------------- training

def train(c, log_every=0):
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    torch.set_num_threads(c["threads"])
    dev = torch.device(c["device"])
    tier, arm, d = c["tier"], c["arm"], c["d"]
    vae = tier == "vae"
    enc = Encoder(d, c["ch"], c["hidden"], out_mult=2 if (vae and arm == "betavae") else 1).to(dev)
    params = list(enc.parameters())
    if vae:
        dec = Decoder(d, c["ch"], c["hidden"]).to(dev); params += list(dec.parameters())
    else:
        pin = d * (c["T"] - 1) if tier == "temporal" else d
        pred = Predictor(pin, d, c["hidden"], norm=c["pred_norm"]).to(dev)
    use_ema = arm == "ema" or arm.startswith("ema+")
    tgt = make_ema(enc) if use_ema else None
    reg_name = arm.split("+")[1] if arm.startswith("ema+") else arm
    if vae:
        reg_name = {"betavae": "none", "ksd": "ksd", "mmd": "mmd"}[arm]
    reg = Regulariser(reg_name, d, law=c["prior"], k=K_CONTENT, num_slices=c["num_slices"],
                      oracle_eps=c["oracle_eps"]).to(dev)
    tnorm = (lambda t: F.layer_norm(t, t.shape[-1:])) if c["ema_target_norm"] == "ln" else (lambda t: t)
    reg.sigreg_impl = c["sigreg_impl"]
    groups = [{"params": params, "lr": c["lr"]}]
    if not vae:
        groups.append({"params": list(pred.parameters()), "lr": c["lr"] * c["pred_lr_mult"]})
    opt = torch.optim.Adam(groups, lr=c["lr"], weight_decay=c["wd"])
    if c["ema_loss"] == "cos":      # BYOL normalised MSE = 2 - 2 cos
        plf = lambda a, b: (2 - 2 * F.cosine_similarity(a, b, dim=-1)).mean()
    else:
        plf = F.mse_loss

    data = Cifar(seed=c["seed"], noise=c["noise"]) if tier == "cifar" else Shapes(c["law"], seed=c["seed"], noise=c["noise"])
    hist = []
    t0 = time.time()
    for step in range(c["steps"]):
        B = c["batch"]
        if tier == "cifar":
            x1, x2, *_ = data.sample_pair(B, c["alpha"], dev)
        elif tier == "temporal":
            f = data.sample_clip_factors(B, c["T"], c["alpha"])
            X = data.images_for(f, dev)                       # (B, T, 3, 32, 32)
        else:
            f1, f2 = data.sample_pair_factors(B, c["alpha"])
            x1, x2 = data.images_for(f1, dev), data.images_for(f2, dev)

        if vae:
            x = x1
            h = enc(x)
            if arm == "betavae":
                mu, logvar = h[:, :d], h[:, d:].clamp(-8, 8)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                r = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean()
            else:
                z = h
                r = reg(z)
            zb = bottleneck(z, c["sigma"])
            rec = F.mse_loss(dec(zb), x, reduction="none").sum((1, 2, 3)).mean()      # SSE per image
            p_loss = rec
        elif tier == "temporal":
            Tn = c["T"]
            z_all = enc(X.flatten(0, 1)).view(B, Tn, d)
            ctx = bottleneck(z_all[:, :-1].reshape(B * (Tn - 1), d), c["sigma"]).view(B, (Tn - 1) * d)
            if use_ema:
                with torch.no_grad():
                    target = tnorm(tgt(X[:, -1]))
            else:
                target = z_all[:, -1]
            p_loss = plf(pred(ctx), target)
            # regulariser per frame (N = B each), averaged over frames, as per view in the image tier
            r = torch.stack([reg(z_all[:, t]) for t in range(Tn)]).mean() if reg_name not in ("none", "ema") \
                else z_all.new_zeros(())
        else:
            z1, z2 = enc(torch.cat([x1, x2])).chunk(2)
            if use_ema:
                with torch.no_grad():
                    t1, t2 = tnorm(tgt(x1)), tnorm(tgt(x2))
            else:
                t1, t2 = z1, z2
            p_loss = 0.5 * (plf(pred(bottleneck(z1, c["sigma"])), t2)
                            + plf(pred(bottleneck(z2, c["sigma"])), t1))
            r = 0.5 * (reg(z1) + reg(z2)) if reg_name not in ("none", "ema") else z1.new_zeros(())
        loss = p_loss + c["lam"] * r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if use_ema:
            ema_update(tgt, enc, c["ema_tau"])
        hist.append(torch.stack([loss.detach(), p_loss.detach(), r.detach()]))   # no host sync per step
        if log_every and (step % log_every == 0 or step == c["steps"] - 1):
            h = hist[-1].tolist()
            print(f"step {step} loss {h[0]:.4f} pred {h[1]:.4f} reg {h[2]:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    hist = torch.stack(hist).float().cpu().numpy()
    train_time = time.time() - t0
    return enc, data, hist, train_time


# ----------------------------------------------------------------------------- evaluation

def evaluate(c, enc, data, hist):
    dev = torch.device(c["device"])
    _embed = embed
    embed_ = lambda e, x, dv: _embed(e, x, dv, dims=c["d"])
    torch.manual_seed(10_000 + c["seed"])
    out = {}
    L = hist[:, 0]
    tail = L[int(0.8 * len(L)):]
    out.update(loss_final=float(tail.mean()), pred_final=float(hist[int(0.8 * len(L)):, 1].mean()),
               reg_final=float(hist[int(0.8 * len(L)):, 2].mean()),
               loss_cv_last20=float(tail.std() / (abs(tail.mean()) + 1e-12)),
               loss_nonfinite=int((~np.isfinite(hist)).any(1).sum()))
    if not np.isfinite(hist).all():
        out["diverged"] = True
    if c["tier"] == "cifar":
        rng = np.random.default_rng(777)
        itr = rng.choice(len(data.ytr), c["n_probe_fit"], replace=False)
        ite = rng.choice(len(data.yte), c["n_probe_eval"], replace=False)
        Ztr = embed_(enc, data.raw(itr, "train") - 0.5, dev)
        Zte = embed_(enc, data.raw(ite, "test") - 0.5, dev)
        out["class_acc"] = M.logreg_acc(Ztr, data.ytr[itr], Zte, data.yte[ite])
        xa, pa = data.augment(data.raw(ite, "test"), c["alpha"] if c["alpha"] > 0 else 1.0)
        Za = embed_(enc, xa, dev)
        h = len(Za) // 2
        out["leak_r2"], per = M.ridge_r2(Za[:h], pa[:h], Za[h:], pa[h:])
        out["leak_r2_per"] = per.tolist()
        Zev = Zte
    else:
        ev = Shapes(c["law"], seed=999, noise=0.0)
        ftr = ev.sample_factors(c["n_probe_fit"], "train")
        fte = ev.sample_factors(c["n_probe_eval"], "test")
        Ztr, Zte = embed_(enc, ev.images_for(ftr, dev, noise=False), dev), embed_(enc, ev.images_for(fte, dev, noise=False), dev)
        Str, Ste, Ntr, Nte = content_code(ftr), content_code(fte), nuisance_code(ftr), nuisance_code(fte)
        out["content_r2"], per = M.ridge_r2(Ztr, Str, Zte, Ste); out["content_r2_per"] = per.tolist()
        out["leak_r2"], per = M.ridge_r2(Ztr, Ntr, Zte, Nte); out["leak_r2_per"] = per.tolist()
        out["content_knn_r2"] = M.knn_r2(Ztr, Str, Zte, Ste)
        out["leak_knn_r2"] = M.knn_r2(Ztr, Ntr, Zte, Nte)
        out["shape_acc"] = M.logreg_acc(Ztr, ftr[:, SH], Zte, fte[:, SH])
        out["ident_orth_r2"] = M.orth_identifiability(Zte, Ste)
        Zev = Zte
    out["eff_rank"] = M.effective_rank(Zev - Zev.mean(0))
    out["emb_std_mean"] = float(Zev.std(0).mean())
    out["gauss_ep_raw"], out["gauss_ep_std"] = M.gaussianity(Zev)
    if np.isfinite(Zev).all() and Zev.std() > 1e-8:
        out["ksd_prior"] = M.ksd_to_prior(Zev, c["prior"])
        out["npksd_prior"], out["npksd_p"] = M.np_ksd(Zev, c["prior"], n=500, n_null=30)
    return out


def run(cfg, outdir="runs", force=False, log_every=0):
    c = resolve(cfg)
    h = cfg_hash(c)
    rd = os.path.join(outdir, h)
    res_path = os.path.join(rd, "result.json")
    if os.path.exists(res_path) and not force:
        return json.load(open(res_path))
    os.makedirs(rd, exist_ok=True)
    enc, data, hist, tt = train(c, log_every)
    t1 = time.time()
    res = evaluate(c, enc, data, hist)
    res.update(config=c, hash=h, train_time_s=tt, eval_time_s=time.time() - t1)
    np.save(os.path.join(rd, "loss_hist.npy"), hist.astype(np.float32))
    tmp = res_path + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, res_path)
    return res


def parse_overrides(args):
    out = {}
    for a in args:
        k, v = a.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") else {}
    rest = sys.argv[2:] if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") else sys.argv[1:]
    cfg.update(parse_overrides(rest))
    r = run(cfg, log_every=200, force=cfg.pop("force", False) if "force" in cfg else False)
    print(json.dumps({k: v for k, v in r.items() if not k.endswith("_per") and k != "config"}, indent=1))
