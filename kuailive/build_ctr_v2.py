# -*- coding: utf-8 -*-
"""
KuaiLive CTR v2: add TIME-AWARE collaborative/historical features (the real CTR lever).

Leakage-safe design:
  - Split the 21-day log by time: history = first HIST_FRAC of the time range,
    label window = the rest. Historical features are computed ONLY from the history
    window; labeled exposures are drawn ONLY from the later window. No leakage.
  - Historical features (smoothed): per-user CTR, per-streamer CTR, and their
    exposure counts (activity), plus content-category CTR.

Then fit LogReg/GBM baselines with STATIC-only vs STATIC+HISTORICAL features to
measure how much the collaborative signal raises the ceiling (free, local).
Outputs ctr_dataset_v2.csv for the subsequent LLM fine-tune.
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("KUAILIVE_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
os.makedirs(OUT, exist_ok=True)

HIST_FRAC = 0.6          # first 60% of time range = history
N_PER_CLASS = 30000      # balanced labeled sample from the later window
ALPHA = 20.0             # Bayesian smoothing strength for CTR
SEED = 42

key = ["user_id", "live_id", "streamer_id", "timestamp"]

print("Loading click + negative ...")
pos = pd.read_csv(os.path.join(BASE, "click.csv"), usecols=key); pos["clicked"] = 1
neg = pd.read_csv(os.path.join(BASE, "negative.csv"), usecols=key); neg["clicked"] = 0
exp = pd.concat([pos, neg], ignore_index=True)
del pos, neg
print(f"  total exposures: {len(exp):,}")

tmin, tmax = exp["timestamp"].min(), exp["timestamp"].max()
cutoff = tmin + HIST_FRAC * (tmax - tmin)
hist = exp[exp["timestamp"] < cutoff]
lab  = exp[exp["timestamp"] >= cutoff]
print(f"  history exposures: {len(hist):,}  | label-window exposures: {len(lab):,}")

g = hist["clicked"].mean()
print(f"  global history CTR: {g:.4f}")

def ctr_table(df, key_col, prefix):
    a = df.groupby(key_col)["clicked"].agg(["sum", "count"])
    a[f"{prefix}_ctr"] = (a["sum"] + ALPHA * g) / (a["count"] + ALPHA)
    a[f"{prefix}_n"] = a["count"]
    return a[[f"{prefix}_ctr", f"{prefix}_n"]]

user_ctr = ctr_table(hist, "user_id", "user")
strm_ctr = ctr_table(hist, "streamer_id", "streamer")

# content-category CTR (join history to room category)
room_cat = pd.read_csv(os.path.join(BASE, "room.csv"),
                       usecols=["live_id", "live_content_category"]).drop_duplicates("live_id")
hist_cat = hist.merge(room_cat, on="live_id", how="left")
cat_ctr = hist_cat.groupby("live_content_category")["clicked"].mean().rename("cat_ctr")
del hist, hist_cat

# balanced labeled sample from the later window
labpos = lab[lab["clicked"] == 1].sample(n=min(N_PER_CLASS, (lab["clicked"] == 1).sum()), random_state=SEED)
labneg = lab[lab["clicked"] == 0].sample(n=min(N_PER_CLASS, (lab["clicked"] == 0).sum()), random_state=SEED)
sample = pd.concat([labpos, labneg], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
del exp, lab, labpos, labneg
print(f"  balanced labeled sample: {len(sample):,}")

# attach historical features
sample = sample.merge(user_ctr, on="user_id", how="left").merge(strm_ctr, on="streamer_id", how="left")
sample["user_ctr"] = sample["user_ctr"].fillna(g); sample["user_n"] = sample["user_n"].fillna(0).astype(int)
sample["streamer_ctr"] = sample["streamer_ctr"].fillna(g); sample["streamer_n"] = sample["streamer_n"].fillna(0).astype(int)

# attach static features (same interpretable set as v1)
user = pd.read_csv(os.path.join(BASE, "user.csv"))
user = user.drop(columns=[c for c in user.columns if c.startswith("onehot_feat")] +
                 [c for c in ["reg_timestamp", "first_watch_live_timestamp"] if c in user.columns])
user = user.rename(columns={c: "u_" + c for c in user.columns if c != "user_id"})
streamer = pd.read_csv(os.path.join(BASE, "streamer.csv"))
streamer = streamer.drop(columns=[c for c in streamer.columns if c.startswith("onehot_feat")] +
                         [c for c in ["first_live_timestamp", "reg_timestamp"] if c in streamer.columns])
streamer = streamer.rename(columns={c: "s_" + c for c in streamer.columns if c != "streamer_id"})

need_live = set(sample["live_id"].unique())
parts = []
for ch in pd.read_csv(os.path.join(BASE, "room.csv"),
                      usecols=["live_id", "live_type", "live_content_category"], chunksize=1_000_000):
    parts.append(ch[ch["live_id"].isin(need_live)])
room = pd.concat(parts, ignore_index=True).drop_duplicates("live_id")
room = room.rename(columns={"live_type": "r_live_type", "live_content_category": "r_content_category"})

df = (sample.merge(user, on="user_id", how="left")
            .merge(streamer, on="streamer_id", how="left")
            .merge(room, on="live_id", how="left"))
df = df.merge(cat_ctr, left_on="r_content_category", right_index=True, how="left")
df["cat_ctr"] = df["cat_ctr"].fillna(g)
ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
df["ctx_hour"] = ts.dt.hour; df["ctx_dayofweek"] = ts.dt.dayofweek

HIST_FEATS = ["user_ctr", "user_n", "streamer_ctr", "streamer_n", "cat_ctr"]
STATIC_FEATS = [c for c in df.columns if c.startswith(("u_", "s_"))] + \
               ["r_content_category", "r_live_type", "ctx_hour", "ctx_dayofweek"]
keep = ["user_id", "live_id", "streamer_id", "clicked"] + HIST_FEATS + STATIC_FEATS
df = df[keep]
out_csv = os.path.join(OUT, "ctr_dataset_v2.csv")
df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"  wrote {out_csv}  ({len(df):,} rows, {len(STATIC_FEATS)} static + {len(HIST_FEATS)} historical feats)")

# ---- baseline comparison: static-only vs static+historical ----
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

n_test = 5000
tr, te = df.iloc[:-n_test], df.iloc[-n_test:]
yte = te["clicked"].values; ytr = tr["clicked"].values

def fit_eval(feat_cols, label):
    Xall = pd.get_dummies(pd.concat([tr[feat_cols], te[feat_cols]], ignore_index=True).astype(
        {c: str for c in feat_cols if df[c].dtype == object}))
    Xtr, Xte = Xall.iloc[:len(tr)], Xall.iloc[len(tr):]
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    print(f"\n[{label}]  ({Xall.shape[1]} encoded cols)")
    for name, m in [("LogReg", lr), ("GBM", gb)]:
        p = m.predict(Xte)
        print(f"   {name:<8}: acc={accuracy_score(yte,p):.4f}  f1={f1_score(yte,p,average='macro'):.4f}")

print("\n================= FEATURE-LEVER TEST (baselines) =================")
fit_eval(STATIC_FEATS, "STATIC only (v1 features)")
fit_eval(STATIC_FEATS + HIST_FEATS, "STATIC + HISTORICAL (v2)")
print(f"\ntest click rate: {yte.mean():.3f}")
