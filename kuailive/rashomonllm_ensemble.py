# -*- coding: utf-8 -*-
"""
Ensemble of the v2 fine-tuned RashomonLLM with the GBM baseline (KuaiLive CTR).

Tests whether the LLM adds COMPLEMENTARY predictive signal: if blending the LLM's
click-probability with the tree's beats both alone, the LLM contributes information
the tree misses. LLM P(click) is read from the chosen-token logprob of the v2 model
(it emits the answer as a single 0/1 token; P(1)=exp(lp) if "1" else 1-exp(lp)).
"""
import os, sys, re, math, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P

DATA = os.path.join(HERE, "analysis", "ctr_dataset_v2.csv")
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
PATH = open(os.path.join(HERE, "analysis", "ft_sampler_path.txt"), encoding="utf-8").read().strip()
FT_TRAIN, FT_TEST, SEED, CONC = 30000, 2000, 42, 8

PROMPT_PREFIX = ("You are predicting user engagement on a live-streaming platform.\n"
                 "Exposure features:\n")
QUESTION = ("\n\nWill this user click into this live room? "
            "Answer with a single digit: 1 = click, 0 = no click.")

def load_tinker_key():
    for line in open(os.path.join(HERE, "tinker_key.local"), encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            if "=" in s and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s.split("=", 1)[0].strip()):
                s = s.split("=", 1)[1]
            os.environ["TINKER_API_KEY"] = s.strip().strip('"').strip("'")
            return

def metrics(y, pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred, average="macro"), 4)}

def main():
    load_tinker_key()
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    P.FEATURES = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]
    train = df.iloc[:FT_TRAIN]; test = df.iloc[FT_TRAIN:FT_TRAIN + FT_TEST].reset_index(drop=True)
    y = test["clicked"].values
    print(f"model={PATH}\ntrain={len(train)} test={len(test)} feats={len(P.FEATURES)}")

    # --- GBM + LogReg probabilities (fit on train) ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    comb = pd.concat([train, test], ignore_index=True)
    num = [c for c in P.FEATURES if pd.api.types.is_numeric_dtype(comb[c]) and comb[c].nunique() > 30]
    cat = [c for c in P.FEATURES if c not in num]
    X = pd.concat([pd.get_dummies(comb[cat].astype(str)), comb[num].reset_index(drop=True)], axis=1)
    Xtr, Xte = X.iloc[:len(train)], X.iloc[len(train):]
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, train["clicked"].values)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, train["clicked"].values)
    p_gbm = gbm.predict_proba(Xte)[:, 1]
    p_lr = lr.predict_proba(Xte)[:, 1]
    print("GBM   :", metrics(y, (p_gbm >= 0.5).astype(int)))
    print("LogReg:", metrics(y, (p_lr >= 0.5).astype(int)))

    # --- LLM probabilities (chosen-token logprob) ---
    from openai import OpenAI
    client = OpenAI(base_url=TINKER_BASE_URL, api_key=os.environ["TINKER_API_KEY"])
    p_llm = np.full(len(test), 0.5)
    def one(i):
        msg = [{"role": "user", "content": PROMPT_PREFIX + P.serialize(test.iloc[i]) + QUESTION}]
        for a in range(4):
            try:
                r = client.chat.completions.create(model=PATH, messages=msg, max_tokens=1,
                                                   temperature=0, logprobs=True, top_logprobs=0)
                ct = r.choices[0].logprobs.content[0]
                tok = ct.token.strip(); pe = math.exp(ct.logprob)
                return i, (pe if tok == "1" else (1 - pe) if tok == "0" else 0.5)
            except Exception:
                time.sleep(2 ** a)
        return i, 0.5
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(test))]):
            i, p = fut.result(); p_llm[i] = p; done += 1
            if done % 250 == 0: print(f"  llm {done}/{len(test)} ({time.time()-t0:.0f}s)")
    print("LLM   :", metrics(y, (p_llm >= 0.5).astype(int)))

    # --- ensemble grid (blend probabilities) ---
    print("\n--- ensemble (w*LLM + (1-w)*GBM) ---")
    best = (-1, None)
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        blend = w * p_llm + (1 - w) * p_gbm
        m = metrics(y, (blend >= 0.5).astype(int))
        print(f"  w={w:.1f}  {m}")
        if m["acc"] > best[0]: best = (m["acc"], w, m)
    print(f"\nBEST ensemble: w={best[1]} {best[2]}")
    # correlation of errors (complementarity signal)
    e_llm = (p_llm >= 0.5).astype(int) != y
    e_gbm = (p_gbm >= 0.5).astype(int) != y
    print(f"error overlap: LLM-wrong={e_llm.mean():.3f} GBM-wrong={e_gbm.mean():.3f} "
          f"both-wrong={(e_llm & e_gbm).mean():.3f} corr={np.corrcoef(e_llm, e_gbm)[0,1]:.3f}")

    rec = {"path": PATH, "gbm": metrics(y, (p_gbm >= 0.5).astype(int)),
           "logreg": metrics(y, (p_lr >= 0.5).astype(int)),
           "llm": metrics(y, (p_llm >= 0.5).astype(int)),
           "best_ensemble": {"w": best[1], **best[2]}}
    json.dump(rec, open(os.path.join(HERE, "analysis", "ensemble_result.json"), "w"), indent=2)
    print("saved ensemble_result.json")

if __name__ == "__main__":
    main()
