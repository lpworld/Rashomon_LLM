# -*- coding: utf-8 -*-
"""
Build an analysis-ready table for the KuaiLive gift-tier classification task
used as the IS-application experiment in the RashomonLLM paper.

Unit of analysis : one (user, live, streamer) gifting session.
Target           : 3-tier total gift value  -> Low (<=5) / Mid (6-99) / High (>=100).
Features         : interpretable, named user / streamer / live-room attributes only.
Dropped          : onehot_feat0-6 (anonymized), title embeddings, raw ids (kept as keys only).

Outputs (in KuaiLive/analysis/):
  gift_tier_dataset.csv  - the clean modeling table
  build_summary.txt      - tier distribution, feature list, missingness, sample rows
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("KUAILIVE_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
os.makedirs(OUT, exist_ok=True)

# Tier thresholds on TOTAL gift value per session (easy to tweak).
LOW_MAX  = 5      # Low : total <= 5
MID_MAX  = 99     # Mid : 6..99 ; High : >= 100

def tier(v):
    if v <= LOW_MAX:
        return "Low"
    if v <= MID_MAX:
        return "Mid"
    return "High"

# ----------------------------------------------------------------------
# 1. Gifts -> aggregate to (user, live, streamer) session
# ----------------------------------------------------------------------
print("Loading gift.csv ...")
gift = pd.read_csv(os.path.join(BASE, "gift.csv"))
print(f"  raw gift events: {len(gift):,}")

agg = (gift.groupby(["user_id", "live_id", "streamer_id"], as_index=False)
            .agg(total_gift_price=("gift_price", "sum"),
                 gift_count=("gift_price", "count"),
                 first_gift_ts=("timestamp", "min")))
agg["gift_tier"] = agg["total_gift_price"].apply(tier)
print(f"  gifting sessions (rows): {len(agg):,}")

# ----------------------------------------------------------------------
# 2. User features (drop anonymized onehot, prefix u_)
# ----------------------------------------------------------------------
print("Loading user.csv ...")
user = pd.read_csv(os.path.join(BASE, "user.csv"))
user = user.drop(columns=[c for c in user.columns if c.startswith("onehot_feat")])
user = user.drop(columns=[c for c in ["reg_timestamp", "first_watch_live_timestamp"]
                          if c in user.columns])
user = user.rename(columns={c: "u_" + c for c in user.columns if c != "user_id"})

# ----------------------------------------------------------------------
# 3. Streamer features (drop anonymized onehot, prefix s_)
# ----------------------------------------------------------------------
print("Loading streamer.csv ...")
streamer = pd.read_csv(os.path.join(BASE, "streamer.csv"))
streamer = streamer.drop(columns=[c for c in streamer.columns if c.startswith("onehot_feat")])
streamer = streamer.drop(columns=[c for c in ["first_live_timestamp", "reg_timestamp"]
                                  if c in streamer.columns])
streamer = streamer.rename(columns={c: "s_" + c for c in streamer.columns
                                    if c != "streamer_id"})

# ----------------------------------------------------------------------
# 4. Room features (chunked filter on needed live_ids; large file)
# ----------------------------------------------------------------------
print("Loading room.csv (chunked filter) ...")
need_live = set(agg["live_id"].unique())
usecols = ["live_id", "live_type", "start_timestamp", "end_timestamp", "live_content_category"]
parts = []
for ch in pd.read_csv(os.path.join(BASE, "room.csv"), usecols=usecols, chunksize=1_000_000):
    parts.append(ch[ch["live_id"].isin(need_live)])
room = pd.concat(parts, ignore_index=True).drop_duplicates(subset="live_id", keep="first")
room["live_duration_min"] = (room["end_timestamp"] - room["start_timestamp"]) / 60000.0
room.loc[room["live_duration_min"] < 0, "live_duration_min"] = np.nan
room = room.rename(columns={"live_type": "r_live_type",
                            "live_content_category": "r_content_category"})
room = room[["live_id", "r_live_type", "r_content_category", "live_duration_min"]]
print(f"  matched live rooms: {len(room):,}")

# ----------------------------------------------------------------------
# 5. Merge + derive context features
# ----------------------------------------------------------------------
df = (agg.merge(user, on="user_id", how="left")
         .merge(streamer, on="streamer_id", how="left")
         .merge(room, on="live_id", how="left"))

ts = pd.to_datetime(df["first_gift_ts"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
df["ctx_hour"] = ts.dt.hour
df["ctx_dayofweek"] = ts.dt.dayofweek  # 0=Mon

# round duration for readability
df["live_duration_min"] = df["live_duration_min"].round(1)

# ----------------------------------------------------------------------
# 6. Final column order: keys -> target -> user -> streamer -> context
# ----------------------------------------------------------------------
keys   = ["user_id", "live_id", "streamer_id"]
target = ["gift_tier", "total_gift_price", "gift_count"]
ucols  = [c for c in df.columns if c.startswith("u_")]
scols  = [c for c in df.columns if c.startswith("s_")]
ctx    = ["r_content_category", "r_live_type", "live_duration_min", "ctx_hour", "ctx_dayofweek"]
df = df[keys + target + ucols + scols + ctx]

out_csv = os.path.join(OUT, "gift_tier_dataset.csv")
df.to_csv(out_csv, index=False, encoding="utf-8")

# ----------------------------------------------------------------------
# 7. Sanity report
# ----------------------------------------------------------------------
lines = []
def emit(s=""):
    print(s)
    lines.append(s)

emit("=" * 64)
emit("KuaiLive gift-tier dataset — build summary")
emit("=" * 64)
emit(f"rows (gifting sessions): {len(df):,}")
emit(f"distinct users        : {df['user_id'].nunique():,}")
emit(f"distinct streamers     : {df['streamer_id'].nunique():,}")
emit(f"distinct lives         : {df['live_id'].nunique():,}")
emit("")
emit(f"tier thresholds: Low total<= {LOW_MAX} | Mid {LOW_MAX+1}-{MID_MAX} | High >= {MID_MAX+1}")
tc = df["gift_tier"].value_counts().reindex(["Low", "Mid", "High"])
emit("tier distribution:")
for k, v in tc.items():
    emit(f"  {k:<5}: {v:>7,}  ({v/len(df):6.1%})")
emit("")
emit(f"total_gift_price : min={df['total_gift_price'].min()}  "
     f"median={df['total_gift_price'].median()}  "
     f"mean={df['total_gift_price'].mean():.1f}  max={df['total_gift_price'].max()}")
emit("")
feat_cols = ucols + scols + ctx
emit(f"feature count (excl. keys+target): {len(feat_cols)}")
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

with open(os.path.join(OUT, "build_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# small preview
print("\n--- sample rows ---")
print(df.head(6).to_string(max_colwidth=18))
