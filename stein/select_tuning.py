"""Select one level per tuning slot from the tuning runs (seeds 100, 101).

    .venv/bin/python -m stein.select_tuning      -> results/tuning_selected.json, results/tuning_table.md

Criterion: highest mean validation linear content R^2 over the tuning seeds (CIFAR: class
accuracy). A level with a non-finite loss in any seed, or with a missing seed, is not eligible.
Leakage is never used for selection, so selection is neutral with respect to H1-H6.
"""
import json, math, os, sys

from .grids import tuning_slots, _levels, _cfg, TUNE_SEEDS, SEL_PATH
from .train import resolve, cfg_hash


def main():
    sel, lines = {}, ["| slot | level | score (mean of seeds) | per-seed | leak_r2 (mean) | chosen |", "|---|---|---|---|---|---|"]
    missing = 0
    for name, over, arm, ctab in tuning_slots():
        metric = "class_acc" if over.get("tier") == "cifar" else "content_r2"
        best = None
        for lv in _levels(arm, ctab):
            vals, leaks, ok = [], [], True
            for sd in TUNE_SEEDS:
                p = os.path.join("runs", cfg_hash(resolve(_cfg(**over, arm=arm, seed=sd, **lv))), "result.json")
                if not os.path.exists(p):
                    ok = False; missing += 1; continue
                r = json.load(open(p))
                if r.get("diverged") or r.get("loss_nonfinite", 0) > 0 or not math.isfinite(r.get(metric, float("nan"))):
                    ok = False
                vals.append(r.get(metric, float("nan"))); leaks.append(r.get("leak_r2", float("nan")))
            score = sum(vals) / len(vals) if vals else float("nan")
            row = dict(level=lv, score=score, per_seed=vals, eligible=ok,
                       leak=sum(leaks) / len(leaks) if leaks else float("nan"))
            if ok and (best is None or score > best["score"]):
                best = row
            lines.append(f"| {name} | {lv} | {score:.3f} | {', '.join(f'{v:.3f}' for v in vals)} | {row['leak']:.3f} | |")
        if best is None:
            print(f"WARNING no eligible level for {name}", file=sys.stderr)
            continue
        sel[name] = dict(level=best["level"], score=best["score"], metric=metric)
        # mark the chosen row
        for i in range(len(lines) - 1, 1, -1):
            if lines[i].startswith(f"| {name} | {best['level']} |"):
                lines[i] = lines[i][:-3] + "| **yes** |"
                break
    os.makedirs("results", exist_ok=True)
    json.dump(sel, open(SEL_PATH, "w"), indent=1)
    open("results/tuning_table.md", "w").write("\n".join(lines) + "\n")
    print(f"selected {len(sel)}/{len(tuning_slots())} slots; {missing} missing tuning runs -> {SEL_PATH}")
    return 0 if len(sel) == len(tuning_slots()) else 1


if __name__ == "__main__":
    sys.exit(main())
