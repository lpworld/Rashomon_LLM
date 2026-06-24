# -*- coding: utf-8 -*-
"""
RashomonLLM EPR pilot on the KuaiLive CTR task.

Goal: a cheap go/no-go signal BEFORE any full run or paper edits.
Answers two questions:
  (1) Does RashomonLLM beat standard local ML baselines (LogReg, GBM)?
  (2) Does the EPR (Reflection / double-loop) step improve over a one-shot explanation?

Cost controls: batched prediction (100 rows/call), small splits, live token/cost
tracking, and a hard MAX_USD stop. Baselines are local (free).

Run:
  set OPENAI_API_KEY=sk-...        (PowerShell:  $env:OPENAI_API_KEY="sk-...")
  python rashomonllm_pilot.py
Optional: set PILOT_MODEL=gpt-4o-mini  to debug the harness cheaply.
"""
import os
import re
import sys
import json
import time
import numpy as np
import pandas as pd

# ----------------------------- config --------------------------------
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "ctr_dataset.csv")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")

def _envi(name, default):  # int env override
    return int(os.environ.get(name, default))

PROVIDER       = os.environ.get("PILOT_PROVIDER", "openai").lower()  # openai | tinker
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
_DEFAULT_MODEL = {"openai": "gpt-4o", "tinker": "Qwen/Qwen3-235B-A22B-Instruct-2507"}
MODEL          = os.environ.get("PILOT_MODEL", _DEFAULT_MODEL.get(PROVIDER, "gpt-4o"))
TRAIN_LEARN_N  = _envi("PILOT_LEARN", 1000)   # labeled rows shown to the Explanation Agent
VALID_N        = _envi("PILOT_VALID", 500)    # validation rows for the EPR loop
TEST_N         = _envi("PILOT_TEST", 2000)    # final held-out test rows
EPR_ITERS      = _envi("PILOT_ITERS", 3)      # Reflection iterations
PRED_BATCH     = _envi("PILOT_PREDBATCH", 50) # rows per prediction call
EXPL_MAX_ROWS  = _envi("PILOT_EXPLROWS", 300) # rows fed to Explanation Agent (context cap)
MAX_ERR_FEED   = 60       # misclassified rows fed to Reflection
MAX_USD        = float(os.environ.get("PILOT_MAX_USD", 8.0))  # hard safety stop
SEED           = 42

# approx pricing $/1M tokens (verify current rates). Tinker runs on your TML credit,
# so we display $0 here and just track token counts.
PRICE = {
    "gpt-4o":      {"in": 2.50, "out": 10.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}
def _price():
    if MODEL in PRICE:
        return PRICE[MODEL]
    return {"in": 0.0, "out": 0.0} if PROVIDER == "tinker" else PRICE["gpt-4o"]

FEATURES = [
    "u_age","u_gender","u_country","u_device_brand","u_device_price","u_fans_num",
    "u_follow_num","u_accu_watch_live_cnt","u_accu_watch_live_duration",
    "u_is_live_streamer","u_is_photo_author","s_gender","s_age","s_country",
    "s_device_brand","s_device_price","s_live_operation_tag","s_fans_user_num",
    "s_fans_group_fans_num","s_follow_user_num","s_accu_live_cnt","s_accu_live_duration",
    "s_accu_play_cnt","s_accu_play_duration","r_content_category","r_live_type",
    "ctx_hour","ctx_dayofweek",
]
TARGET = "clicked"

# --------------------------- llm backend -----------------------------
from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
_KEY_FILES = {"openai": "openai_key.local", "tinker": "tinker_key.local"}
_KEY_ENV   = {"openai": "OPENAI_API_KEY",   "tinker": "TINKER_API_KEY"}
KEY_FILE = os.path.join(_HERE, _KEY_FILES.get(PROVIDER, "openai_key.local"))
ENV_NAME = _KEY_ENV.get(PROVIDER, "OPENAI_API_KEY")

def _load_key():
    """Use the provider's env key if set; otherwise read it from the local key file.
    File may contain a bare key or a `NAME=value` line; blanks and #comments ignored."""
    if os.environ.get(ENV_NAME):
        return
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    name, val = line.split("=", 1)
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name.strip()):
                        line = val
                line = line.strip().strip('"').strip("'")
                if line:
                    os.environ[ENV_NAME] = line
                    return

_client = None
def client():
    global _client
    if _client is None:
        _load_key()
        key = os.environ.get(ENV_NAME)
        if not key:
            sys.exit(f"ERROR: no {ENV_NAME}. Put your key in {KEY_FILE} (or set {ENV_NAME}).")
        kwargs = {"api_key": key}
        if PROVIDER == "tinker":
            kwargs["base_url"] = TINKER_BASE_URL
        _client = OpenAI(**kwargs)
    return _client

class Cost:
    def __init__(self): self.tin = 0; self.tout = 0; self.calls = 0
    def add(self, u):
        self.tin += u.prompt_tokens; self.tout += u.completion_tokens; self.calls += 1
    def usd(self):
        p = _price()
        return self.tin/1e6*p["in"] + self.tout/1e6*p["out"]
    def line(self):
        return (f"[cost] calls={self.calls} in={self.tin:,} out={self.tout:,} "
                f"~${self.usd():.2f}")
COST = Cost()

def chat(system, user, max_tokens=1500, temperature=0.0):
    if COST.usd() > MAX_USD:
        sys.exit(f"STOP: spend ~${COST.usd():.2f} exceeded MAX_USD=${MAX_USD}.")
    for attempt in range(5):
        try:
            r = client().chat.completions.create(
                model=MODEL, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user}])
            COST.add(r.usage)
            return r.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  api error ({e}); retry in {wait}s")
            time.sleep(wait)
    sys.exit("ERROR: repeated API failures.")

# ----------------------------- data ----------------------------------
_PREFIX = {"u_": "user.", "s_": "streamer.", "r_": "room.", "ctx_": "ctx."}
def _pretty(f):
    for p, repl in _PREFIX.items():
        if f.startswith(p):
            return repl + f[len(p):]
    return f

def serialize(row, idx=None):
    body = " | ".join(f"{_pretty(f)}={row[f]}" for f in FEATURES)
    return (f"[{idx}] {body}" if idx is not None else body)

def load_splits():
    df = pd.read_csv(DATA)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # disjoint stratified-ish splits (data is already 50/50 balanced & shuffled)
    learn = df.iloc[:TRAIN_LEARN_N]
    valid = df.iloc[TRAIN_LEARN_N:TRAIN_LEARN_N+VALID_N]
    test  = df.iloc[TRAIN_LEARN_N+VALID_N:TRAIN_LEARN_N+VALID_N+TEST_N]
    return df, learn.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

# ----------------------------- agents --------------------------------
EXPL_SYS = ("You are an expert data scientist analyzing a live-streaming platform. "
            "You discover and articulate the conditional feature patterns that explain "
            "whether a user CLICKS into a streamer's live room (clicked=1) or not (clicked=0). "
            "Your description will be used to predict clicks on unseen exposures, so be "
            "specific, conditional, and grounded in the examples.")

def explanation_agent(learn):
    sample = learn.head(EXPL_MAX_ROWS)  # cap rows so we stay well under the context window
    ex = "\n".join(f"[clicked={r[TARGET]}] {serialize(r)}" for _, r in sample.iterrows())
    user = ("Below are labeled exposure examples. Identify the key conditional patterns "
            "that distinguish clicked=1 from clicked=0 (e.g., interactions among user "
            "engagement history, streamer content niche/popularity, and time-of-day). "
            "Return a concise, structured set of rules/relationships (<= 350 words).\n\n"
            f"{ex}")
    return chat(EXPL_SYS, user, max_tokens=900)

PRED_SYS = ("You are a precise click-prediction engine. Given a learned description of "
            "click patterns and a batch of exposures, predict clicked (0 or 1) for EACH row. "
            "Output ONLY lines of the form `idx=0` or `idx=1`, one per row, no other text.")

def predict_batch(explanation, rows):
    preds = np.full(len(rows), -1, dtype=int)
    for start in range(0, len(rows), PRED_BATCH):
        chunk = rows.iloc[start:start+PRED_BATCH]
        n = len(chunk)
        # local indices 0..n-1 within the batch (short tokens, remapped to global)
        block = "\n".join(serialize(r, idx=i) for i,(_,r) in enumerate(chunk.iterrows()))
        user = (f"LEARNED CLICK PATTERNS:\n{explanation}\n\n"
                f"Predict clicked (0 or 1) for each of the {n} rows below. Output EXACTLY "
                f"{n} lines, each `i=0` or `i=1` where i is the bracket number. No other text.\n\n"
                f"{block}")
        out = chat(PRED_SYS, user, max_tokens=n*10 + 100)
        for m in re.finditer(r"(\d+)\s*=\s*([01])", out):
            li = int(m.group(1))
            if 0 <= li < n:
                preds[start+li] = int(m.group(2))
    miss = int((preds == -1).sum())
    preds[preds == -1] = 0   # default unparsed to 0; tracked via miss
    return preds, miss

REFL_SYS = ("You are a reflective analyst performing double-loop learning. You examine "
            "systematic prediction errors and REVISE the click-pattern description so it "
            "better matches reality. Output ONLY the improved description.")

def reflection_agent(explanation, valid, preds):
    wrong = valid[(valid[TARGET].values != preds)]
    wrong = wrong.head(MAX_ERR_FEED)
    if len(wrong) == 0:
        return explanation
    err = "\n".join(f"[true clicked={r[TARGET]}, predicted={p}] {serialize(r)}"
                    for (_, r), p in zip(wrong.iterrows(),
                                         preds[(valid[TARGET].values != preds)][:MAX_ERR_FEED]))
    user = (f"CURRENT DESCRIPTION:\n{explanation}\n\n"
            f"These exposures were MISPREDICTED (with true vs predicted labels). Diagnose the "
            f"systematic gaps and rewrite an improved description (<= 350 words).\n\n{err}")
    return chat(REFL_SYS, user, max_tokens=900)

# ----------------------------- metrics -------------------------------
from sklearn.metrics import accuracy_score, f1_score

def score(y_true, y_pred):
    return {"acc": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "f1_pos": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4)}

def local_baselines(full, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    sub = full.iloc[:TRAIN_LEARN_N+VALID_N+TEST_N].copy()
    X = pd.get_dummies(sub[FEATURES].astype(str), drop_first=False)
    y = sub[TARGET].values
    n_tr = TRAIN_LEARN_N + VALID_N
    Xtr, ytr = X.iloc[:n_tr], y[:n_tr]
    Xte, yte = X.iloc[n_tr:n_tr+TEST_N], y[n_tr:n_tr+TEST_N]
    res = {}
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    res["LogisticRegression"] = score(yte, lr.predict(Xte))
    gb = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    res["HistGradientBoosting"] = score(yte, gb.predict(Xte))
    return res, yte

# ----------------------------- main ----------------------------------
def main():
    print(f"Model={MODEL}  learn={TRAIN_LEARN_N} valid={VALID_N} test={TEST_N} "
          f"iters={EPR_ITERS}  MAX_USD=${MAX_USD}")
    full, learn, valid, test = load_splits()
    y_test = test[TARGET].values

    print("\n[1] Local baselines (free)...")
    base, _ = local_baselines(full, test)
    for k, v in base.items(): print(f"   {k:<22}: {v}")

    print("\n[2] Explanation Agent (initial)...")
    expl = explanation_agent(learn)
    print("   " + COST.line())

    print("\n[3] One-shot test (no reflection)...")
    p0, m0 = predict_batch(expl, test)
    s_oneshot = score(y_test, p0)
    print(f"   one-shot: {s_oneshot}  (unparsed rows defaulted: {m0})")
    print("   " + COST.line())

    print("\n[4] EPR loop on validation...")
    traj = []
    for it in range(1, EPR_ITERS+1):
        pv, mv = predict_batch(expl, valid)
        sv = score(valid[TARGET].values, pv)
        traj.append(sv)
        print(f"   iter {it}: valid {sv}  (misses {mv})  " + COST.line())
        expl = reflection_agent(expl, valid, pv)

    print("\n[5] Final test (EPR-refined explanation)...")
    pf, mf = predict_batch(expl, test)
    s_final = score(y_test, pf)
    print(f"   EPR-final: {s_final}  (unparsed rows defaulted: {mf})")

    print("\n================= PILOT RESULT =================")
    print(f"  Baseline LogReg     : {base['LogisticRegression']}")
    print(f"  Baseline GBM        : {base['HistGradientBoosting']}")
    print(f"  RashomonLLM one-shot: {s_oneshot}")
    print(f"  RashomonLLM EPR     : {s_final}")
    print(f"  EPR lift (acc)      : {round(s_final['acc']-s_oneshot['acc'],4)}")
    print("  " + COST.line())

    rec = {"model": MODEL, "config": {"learn": TRAIN_LEARN_N, "valid": VALID_N,
            "test": TEST_N, "iters": EPR_ITERS},
           "baselines": base, "oneshot": s_oneshot, "epr_final": s_final,
           "epr_valid_trajectory": traj,
           "final_explanation": expl,
           "cost": {"calls": COST.calls, "in": COST.tin, "out": COST.tout,
                    "usd_est": round(COST.usd(), 3)}}
    with open(os.path.join(OUT, "pilot_result.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {os.path.join(OUT, 'pilot_result.json')}")

if __name__ == "__main__":
    main()
