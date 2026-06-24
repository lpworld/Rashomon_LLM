# -*- coding: utf-8 -*-
"""
Bootstrap standard errors for the KuaiLive table numbers (single-run test-set bootstrap).
Reports mean +/- std for: prediction accuracy/F1 (RashomonLLM, LogReg, GBM) and
explanation single-deletion drop@5 / stability (RashomonLLM, SparseTree, Permutation, Random).
"""
import os, sys, re, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P

DATA = os.path.join(HERE, "analysis", "ctr_dataset_v2.csv")
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
FT_PATH = open(os.path.join(HERE, "analysis", "ft_sampler_path.txt"), encoding="utf-8").read().strip()
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FT_TRAIN, FT_TEST, SEED, CONC, B = 30000, 2000, 42, 8, 2000
PROMPT_PREFIX = ("You are predicting user engagement on a live-streaming platform.\n"
                 "Exposure features:\n")
QUESTION = ("\n\nWill this user click into this live room? "
            "Answer with a single digit: 1 = click, 0 = no click.")
FEATURE_GLOSS = {"user_ctr": "user's historical click-through rate", "user_n": "user's past exposure count",
    "streamer_ctr": "streamer's historical CTR", "streamer_n": "streamer's past exposure count",
    "cat_ctr": "content-category historical CTR"}

def load_tinker_key():
    for line in open(os.path.join(HERE, "tinker_key.local"), encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            if "=" in s and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s.split("=", 1)[0].strip()):
                s = s.split("=", 1)[1]
            os.environ["TINKER_API_KEY"] = s.strip().strip('"').strip("'"); return

def acc_vec(y, p): return (y == p).astype(float)
def f1_boot(y, p, idx):
    from sklearn.metrics import f1_score
    return f1_score(y[idx], p[idx], average="macro")

def main():
    load_tinker_key()
    from openai import OpenAI
    client = OpenAI(base_url=TINKER_BASE_URL, api_key=os.environ["TINKER_API_KEY"])
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    FEATS = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]
    P.FEATURES = FEATS
    train = df.iloc[:FT_TRAIN].reset_index(drop=True)
    test = df.iloc[FT_TRAIN:FT_TRAIN + FT_TEST].reset_index(drop=True)
    ytr, yte = train["clicked"].values, test["clicked"].values

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.inspection import permutation_importance
    num = [c for c in FEATS if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 30]
    cat = [c for c in FEATS if c not in num]
    Xtr = pd.concat([pd.get_dummies(train[cat].astype(str)), train[num].reset_index(drop=True)], axis=1)
    COLS = Xtr.columns
    def encode(frame):
        X = pd.concat([pd.get_dummies(frame[cat].astype(str)), frame[num].reset_index(drop=True)], axis=1)
        return X.reindex(columns=COLS, fill_value=0)
    def ablate(frame, kill):
        t = frame.copy()
        for f in kill: t[f] = train[f].mean() if f in num else train[f].mode().iloc[0]
        return t
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    Xte = encode(test)
    p_gbm = gbm.predict(Xte); p_lr = lr.predict(Xte); base_pred = p_gbm

    # LLM test predictions
    print("getting LLM test predictions...")
    p_llm = np.zeros(len(test), dtype=int)
    def one(i):
        msg = [{"role": "user", "content": PROMPT_PREFIX + P.serialize(test.iloc[i]) + QUESTION}]
        for a in range(4):
            try:
                r = client.chat.completions.create(model=FT_PATH, messages=msg, max_tokens=1, temperature=0)
                m = re.search(r"[01]", r.choices[0].message.content or ""); return i, (int(m.group()) if m else 0)
            except Exception: time.sleep(2 ** a)
        return i, 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(test))]):
            i, p = fut.result(); p_llm[i] = p

    # rankings (for explanation metrics)
    pi = permutation_importance(gbm, Xte, yte, n_repeats=5, random_state=SEED, n_jobs=-1)
    cimp = dict(zip(COLS, pi.importances_mean))
    def agg(d, f): return d.get(f, 0.0) if f in num else sum(v for c, v in d.items() if c.startswith(f + "_"))
    rk = {"Permutation": sorted(FEATS, key=lambda f: -agg(cimp, f))}
    dt = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(Xtr, ytr)
    dcol = dict(zip(COLS, dt.feature_importances_))
    rk["SparseTree"] = sorted(FEATS, key=lambda f: -agg(dcol, f))
    rng = np.random.RandomState(SEED); rnd = FEATS.copy(); rng.shuffle(rnd); rk["Random"] = rnd
    feats_desc = "\n".join(f"- {f}" + (f" ({FEATURE_GLOSS[f]})" if f in FEATURE_GLOSS else "") for f in FEATS)
    txt = client.chat.completions.create(model=BASE_MODEL, max_tokens=700, temperature=0.0,
        messages=[{"role": "user", "content": "Rank ALL these features from MOST to LEAST important for "
                   "predicting whether a user clicks into a live room. Output only feature names, one per line, "
                   "most important first, exact names.\n\n" + feats_desc}]).choices[0].message.content or ""
    rk["RashomonLLM"] = sorted(FEATS, key=lambda f: (re.search(re.escape(f), txt).start() if re.search(re.escape(f), txt) else 10**9))

    # per-row vectors for explanation metrics
    base_correct = acc_vec(yte, base_pred)
    expl = {}
    for name, r in rk.items():
        del_correct = acc_vec(yte, gbm.predict(encode(ablate(test, r[:5]))))   # drop@5 = base - ablated
        stab_agree = (gbm.predict(encode(ablate(test, r[-10:]))) == base_pred).astype(float)
        expl[name] = (del_correct, stab_agree)

    # bootstrap
    rs = np.random.RandomState(SEED); n = len(test)
    pred_metrics = {"RashomonLLM": p_llm, "LogReg": p_lr, "GBM": p_gbm}
    acc_b = {k: [] for k in pred_metrics}; f1_b = {k: [] for k in pred_metrics}
    drop_b = {k: [] for k in expl}; stab_b = {k: [] for k in expl}
    for _ in range(B):
        idx = rs.randint(0, n, n)
        for k, p in pred_metrics.items():
            acc_b[k].append((yte[idx] == p[idx]).mean()); f1_b[k].append(f1_boot(yte, p, idx))
        for k, (dc, sa) in expl.items():
            drop_b[k].append(base_correct[idx].mean() - dc[idx].mean()); stab_b[k].append(sa[idx].mean())

    def ms(a): return f"{np.mean(a):.4f} +/- {np.std(a):.4f}"
    print("\n=========== PREDICTION (mean +/- bootstrap std, n=2000) ===========")
    for k in pred_metrics: print(f"  {k:<12} acc {ms(acc_b[k])}   f1 {ms(f1_b[k])}")
    print("\n=========== EXPLANATION QUALITY (mean +/- bootstrap std) ===========")
    for k in ["RashomonLLM", "SparseTree", "Permutation", "Random"]:
        print(f"  {k:<12} drop@5 {ms(drop_b[k])}   stability {ms(stab_b[k])}")

    out = {"pred": {k: {"acc_mean": float(np.mean(acc_b[k])), "acc_std": float(np.std(acc_b[k])),
                        "f1_mean": float(np.mean(f1_b[k])), "f1_std": float(np.std(f1_b[k]))} for k in pred_metrics},
           "expl": {k: {"drop_mean": float(np.mean(drop_b[k])), "drop_std": float(np.std(drop_b[k])),
                        "stab_mean": float(np.mean(stab_b[k])), "stab_std": float(np.std(stab_b[k]))} for k in expl}}
    json.dump(out, open(os.path.join(HERE, "analysis", "std_result.json"), "w"), indent=2)
    print("\nsaved std_result.json")

if __name__ == "__main__":
    main()
