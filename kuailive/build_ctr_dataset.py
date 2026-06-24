# -*- coding: utf-8 -*-
"""
Build an analysis-ready table for the KuaiLive CTR (click-through-rate)
classification task used as the IS-application experiment in the RashomonLLM paper.

Unit of analysis : one (user, live, streamer) exposure.
Label            : clicked = 1 (from click.csv)  /  0 (real exposed non-click, negative.csv).
Features         : interpretable, named user / streamer / live-room attributes only
                   (pre-exposure context; NO watch_live_time / live duration -> no leakage).
Dropped          : onehot_feat0-6, title embeddings, raw ids (kept as keys only),
                   post-click outcomes.

Strategy         : report FULL-scale counts (~17.6M), then draw a class-balanced
                   modeling SAMPLE before joining features (keeps output small + cheap).

Outputs (in KuaiLive/analysis/):
  ctr_dataset.csv      - the balanced modeling table
  ctr_build_summary.txt - scale, class balance, feature list, missingness, sample rows
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("KUAILIVE_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
os.makedirs(OUT, exist_ok=True)

SAMPLE_PER_CLASS = 250_000   # -> 500k balanced rows; tweak for cost/size
SEED = 42

key_cols = ["user_id", "live_id", "streamer_id", "timestamp"]

# ----------------------------------------------------------------------
# 1. Positives (click) + Negatives (real exposed non-click)
# ----------------------------------------------------------------------
print("Loading click.csv (positives) ...")
pos = pd.read_csv(os.path.join(BASE, "click.csv"), usecols=key_cols)
n_pos = len(pos)
print(f"  positive exposures (clicks): {n_pos:,}")

print("Loading negative.csv (negatives) ...")
neg = pd.read_csv(os.path.join(BASE, "negative.csv"), usecols=key_cols)
n_neg = len(neg)
print(f"  negative exposures         : {n_neg:,}")

n_total = n_pos + n_neg
print(f"  FULL interaction log       : {n_total:,} exposures")

# ----------------------------------------------------------------------
# 2. Balanced sample BEFORE joining features (cheap + small output)
# ----------------------------------------------------------------------
n_each = min(SAMPLE_PER_CLASS, len(pos), len(neg))
pos_s = pos.sample(n=n_each, random_state=SEED).copy()
neg_s = neg.sample(n=n_each, random_state=SEED).copy()
pos_s["clicked"] = 1
neg_s["clicked"] = 0
sample = pd.concat([pos_s, neg_s], ignore_index=True)
sample = sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle
del pos, neg, pos_s, neg_s
print(f"  balanced modeling sample   : {len(sample):,} ({n_each:,}/class)")

# ----------------------------------------------------------------------
# 3. User features (drop anonymized onehot + timestamps, prefix u_)
# ----------------------------------------------------------------------
print("Loading user.csv ...")
user = pd.read_csv(os.path.join(BASE, "user.csv"))
user = user.drop(columns=[c for c in user.columns if c.startswith("onehot_feat")])
user = user.drop(columns=[c for c in ["reg_timestamp", "first_watch_live_timestamp"]
                          if c in user.columns])
user = user.rename(columns={c: "u_" + c for c in user.columns if c != "user_id"})

# ----------------------------------------------------------------------
# 4. Streamer features (drop anonymized onehot + timestamps, prefix s_)
# ----------------------------------------------------------------------
print("Loading streamer.csv ...")
streamer = pd.read_csv(os.path.join(BASE, "streamer.csv"))
streamer = streamer.drop(columns=[c for c in streamer.columns if c.startswith("onehot_feat")])
streamer = streamer.drop(columns=[c for c in ["first_live_timestamp", "reg_timestamp"]
                                  if c in streamer.columns])
streamer = streamer.rename(columns={c: "s_" + c for c in streamer.columns
                                    if c != "streamer_id"})

# ----------------------------------------------------------------------
# 5. Room features (chunked filter on sampled live_ids; content + type only,
#    NO duration -> avoid post-exposure leakage)
# ----------------------------------------------------------------------
print("Loading room.csv (chunked filter) ...")
need_live = set(sample["live_id"].unique())
usecols = ["live_id", "live_type", "live_content_category"]
parts = []
for ch in pd.read_csv(os.path.join(BASE, "room.csv"), usecols=usecols, chunksize=1_000_000):
    parts.append(ch[ch["live_id"].isin(need_live)])
room = pd.concat(parts, ignore_index=True).drop_duplicates(subset="live_id", keep="first")
room = room.rename(columns={"live_type": "r_live_type",
                            "live_content_category": "r_content_category"})
print(f"  matched live rooms: {len(room):,}")

# ----------------------------------------------------------------------
# 6. Merge + derive non-leaky context features
# ----------------------------------------------------------------------
df = (sample.merge(user, on="user_id", how="left")
            .merge(streamer, on="streamer_id", how="left")
            .merge(room, on="live_id", how="left"))

ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
df["ctx_hour"] = ts.dt.hour
df["ctx_dayofweek"] = ts.dt.dayofweek  # 0=Mon

# ----------------------------------------------------------------------
# 7. Final column order: keys -> label -> user -> streamer -> context
# ----------------------------------------------------------------------
keys   = ["user_id", "live_id", "streamer_id"]
label  = ["clicked"]
ucols  = [c for c in df.columns if c.startswith("u_")]
scols  = [c for c in df.columns if c.startswith("s_")]
ctx    = ["r_content_category", "r_live_type", "ctx_hour", "ctx_dayofweek"]
df = df[keys + label + ucols + scols + ctx]

out_csv = os.path.join(OUT, "ctr_dataset.csv")
df.to_csv(out_csv, index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# 8. Sanity report
# ----------------------------------------------------------------------
lines = []
def emit(s=""):
    print(s)
    lines.append(s)

emit("=" * 64)
emit("KuaiLive CTR dataset — build summary")
emit("=" * 64)
emit(f"FULL interaction log : {n_total:,} exposures")
emit(f"  full positives (clicks)   : {n_pos:,}")
emit(f"  full negatives (exposed)  : {n_neg:,}")
emit(f"  natural CTR (pos/total)   : {n_pos / n_total:.3%}")
emit(f"modeling sample rows : {len(df):,}  ({n_each:,}/class, balanced)")
emit(f"distinct users        : {df['user_id'].nunique():,}")
emit(f"distinct streamers     : {df['streamer_id'].nunique():,}")
emit(f"distinct lives         : {df['live_id'].nunique():,}")
emit("")
bal = df["clicked"].value_counts().reindex([1, 0])
emit("class balance:")
for k, v in bal.items():
    emit(f"  clicked={k}: {v:>7,}  ({v/len(df):6.1%})")
emit("")
feat_cols = ucols + scols + ctx
emit(f"feature count (excl. keys+label): {len(feat_cols)}")
emit("features: " + ", ".join(feat_cols))
emit("")
emit("missingness (cols with any NaN):")
miss = df[feat_cols].isna().mean()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) == 0:
    emit("  none")
else:
    for c, m in miss.items():
        emit(f"  {c:<28}: {m:.2%}")
emit("")
emit("content category distribution:")
for k, v in df["r_content_category"].value_counts(dropna=False).items():
    emit(f"  {str(k):<10}: {v:>7,}")
emit("")
emit(f"output written: {out_csv}")

with open(os.path.join(OUT, "ctr_build_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n--- sample rows ---")
print(df.head(6).to_string(max_colwidth=18))
