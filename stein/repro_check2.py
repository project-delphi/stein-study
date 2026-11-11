"""Compare re-run results in a fresh run cache with the stored ones in runs/ (study 2, REPRODUCE2.md).

    STEIN_RUNS=/tmp/s2_fresh .venv/bin/python -W ignore -m stein.sweep s2_fig_e4 --workers 6
    .venv/bin/python -m stein.repro_check2 /tmp/s2_fresh

For every runs/<hash>/result.json present in the fresh directory, prints the largest absolute
difference over the scalar metrics against runs/<hash>/result.json, and exits 1 if any exceeds --tol.
"""
import argparse, glob, json, os, sys

SKIP = {"train_time_s", "eval_time_s", "device", "threads", "hash"}


def scalars(d, prefix=""):
    for k, v in d.items():
        if k in SKIP:
            continue
        if isinstance(v, dict):
            yield from scalars(v, prefix + k + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield prefix + k, float(v)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fresh")
    ap.add_argument("--ref", default="runs")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args(argv)
    worst, n = 0.0, 0
    for f in sorted(glob.glob(os.path.join(a.fresh, "*", "result.json"))):
        h = os.path.basename(os.path.dirname(f))
        ref = os.path.join(a.ref, h, "result.json")
        if not os.path.exists(ref):
            print(f"{h}: no stored result"); continue
        new, old = dict(scalars(json.load(open(f)))), dict(scalars(json.load(open(ref))))
        diffs = {k: abs(new[k] - old[k]) for k in new.keys() & old.keys()}
        k = max(diffs, key=diffs.get) if diffs else None
        d = diffs.get(k, 0.0)
        worst, n = max(worst, d), n + 1
        print(f"{h}: {len(diffs)} metrics, max |diff| = {d:.3g}" + (f" ({k})" if d > 0 else ""))
    print(f"{n} runs compared; largest |diff| = {worst:.3g} (tol {a.tol:g})")
    sys.exit(0 if n and worst <= a.tol else 1)


if __name__ == "__main__":
    main()
