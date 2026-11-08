# Summarise pilot runs (results/pilot_*.parquet) into results/pilot_summary.md for GATE_1.
import glob, pandas as pd
rows = []
for f in sorted(glob.glob("results/pilot_*.parquet")):
    df = pd.read_parquet(f); df["file"] = f.split("/")[-1]; rows.append(df)
df = pd.concat(rows, ignore_index=True)
cols = ["file", "tier", "arm", "lam", "ch", "steps", "law", "prior", "T", "ema_tau", "pred_lr_mult", "content_r2",
        "content_knn_r2", "class_acc", "leak_r2", "leak_knn_r2", "ident_orth_r2", "eff_rank", "emb_std_mean", "train_time_s"]
cols = [c for c in cols if c in df.columns]
open("results/pilot_summary.md", "w").write("```\n" + df[cols].round(3).to_string(index=False) + "\n```\n")
# effective rate per pilot file: wall time from logs is in logs/pilot_round*.log; per-run train time here
g = df.groupby(["file", "tier"])["train_time_s"].agg(["count", "mean", "max"]).round(1)
print(g.to_string())
