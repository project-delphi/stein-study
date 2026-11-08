"""Grid runner: multiprocessing over configs, one torch thread per worker, idempotent
(finished runs are skipped via runs/<hash>/result.json). Prints PROGRESS lines.

    .venv/bin/python -m stein.sweep <grid.yaml> [--workers N] [--limit N] [--out results/x.parquet]

grid.yaml: {base: {...}, runs: [ {...overrides...}, ... ]}  (runs may also be produced by
stein/grids.py: `--grid <name>`).
"""
import argparse, json, os, sys, time, traceback
import multiprocessing as mp
import yaml

from .train import resolve, cfg_hash, run


def _worker(cfg):
    import torch
    torch.set_num_threads(1)
    t = time.time()
    try:
        r = run(cfg)
        return cfg_hash(resolve(cfg)), True, time.time() - t, None
    except Exception:
        return cfg_hash(resolve(cfg)), False, time.time() - t, traceback.format_exc()


def collect(cfgs, out):
    import pandas as pd
    rows = []
    for c in cfgs:
        rc = resolve(c)
        p = os.path.join("runs", cfg_hash(rc), "result.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        row = {k: v for k, v in r["config"].items()}
        for k, v in r.items():
            if k == "config":
                continue
            row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        rows.append(row)
    if rows and out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.grid.endswith(".yaml"):
        g = yaml.safe_load(open(a.grid))
        cfgs = [{**g.get("base", {}), **r} for r in g["runs"]]
    else:
        from . import grids
        cfgs = getattr(grids, a.grid)()
    if a.limit:
        cfgs = cfgs[: a.limit]
    todo = [c for c in cfgs if not os.path.exists(os.path.join("runs", cfg_hash(resolve(c)), "result.json"))]
    n_all, n_todo = len(cfgs), len(todo)
    print(f"PROGRESS {n_all - n_todo}/{n_all} runs done at start; {n_todo} to run with {a.workers} workers", flush=True)
    t0, done, fails, last = time.time(), 0, 0, time.time()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, maxtasksperchild=8) as pool:
            for h, ok, dt, err in pool.imap_unordered(_worker, todo):
                done += 1
                if not ok:
                    fails += 1
                    print(f"FAIL {h}\n{err}", flush=True)
                el = time.time() - t0
                if time.time() - last > 60 or done == n_todo:
                    rate = el / done
                    print(f"PROGRESS {n_all - n_todo + done}/{n_all} runs, {fails} failed, "
                          f"~{rate * (n_todo - done) / 3600:.2f} h left ({rate:.0f} s/run effective)", flush=True)
                    last = time.time()
    rows = collect(cfgs, a.out)
    print(f"collected {len(rows)} results -> {a.out or '(not saved)'}; failures {fails}", flush=True)
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
