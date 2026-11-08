# Prints a compact table of a results parquet (pilot inspection).

import pandas as pd, sys
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
df = pd.read_parquet(sys.argv[1])
cols = [c for c in ["tier","arm","d","lam","sigma","alpha","law","prior","steps","content_r2","leak_r2","shape_acc","class_acc","ident_orth_r2","eff_rank","emb_std_mean","gauss_ep_raw","ksd_prior","npksd_p","pred_final","reg_final","loss_cv_last20","train_time_s","eval_time_s"] if c in df.columns]
extra = sys.argv[2].split(",") if len(sys.argv) > 2 else []
print(df[cols + extra].round(3).to_string(index=False))
