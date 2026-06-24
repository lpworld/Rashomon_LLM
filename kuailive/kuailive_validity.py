# -*- coding: utf-8 -*-
"""
Ablation studies + robustness checks for the KuaiLive CTR findings (validity battery).
All LLM results use the EXISTING fine-tuned v2 model (no retraining); baselines are local.

ABLATIONS
  1. Feature-group ablation: static -> +user/streamer CTR -> full (collaborative features).
ROBUSTNESS (existing FT model)
  2. Inference temperature: T=0 vs T=1 (prediction stability).
  3. Feature-order permutation: shuffle serialization order (no positional artifact).
  4. Prior shift: evaluate the balanced-trained model on a natural ~28%-CTR test set.
  5. Discrimination: ROC-AUC / PR-AUC for RashomonLLM vs baselines (LLM prob via logprob).

Saves analysis/validity_result.json.
"""
import os, sys, re, math, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P

DATA = os.path.join(HERE, "analysis", "ctr_dataset_v2.csv")
TINKER = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
FT_PATH = open(os.path.join(HERE, "analysis", "ft_sampler_path.txt"), encoding="utf-8").read().strip()
FT_TRAIN, FT_TEST, SEED, CONC = 30000, 2000, 42, 8
NAT_RATE, NAT_N = 0.279, 2000
COLLAB = ["user_ctr", "user_n", "streamer_ctr", "streamer_n", "cat_ctr"]
PROMPT_PREFIX = ("You are predicting user engagement on a live-streaming platform.\nExposure features:\n")
QUESTION = ("\n\nWill this user click into this live room? "
            "Answer with a single digit: 1 = click, 0 = no click.")

def load_key():
    for line in open(os.path.join(HERE, "tinker_key.local"), encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            if "=" in s and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s.split("=", 1)[0].strip()):
                s = s.split("=", 1)[1]
            os.environ["TINKER_API_KEY"] = s.strip().strip('"').strip("'"); return

def main():
    load_key()
    from openai import OpenAI
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                                 roc_auc_score, average_precision_score)
    client = OpenAI(base_url=TINKER, api_key=os.environ["TINKER_API_KEY"])
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    ALL = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]
    P.FEATURES = ALL
    train = df.iloc[:FT_TRAIN].reset_index(drop=True)
    test = df.iloc[FT_TRAIN:FT_TRAIN + FT_TEST].reset_index(drop=True)
    pool = df.iloc[FT_TRAIN + FT_TEST:].reset_index(drop=True)
    ytr, yte = train["clicked"].values, test["clicked"].values
    out = {}

    # ---------- baseline encoder for an arbitrary feature subset ----------
    def fit_eval(feats):
        num = [c for c in feats if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 30]
        cat = [c for c in feats if c not in num]
        def enc(frame, cols=None):
            X = pd.concat([pd.get_dummies(frame[cat].astype(str)), frame[num].reset_index(drop=True)], axis=1)
            return X.reindex(columns=cols, fill_value=0) if cols is not None else X
        Xtr = enc(train); cols = Xtr.columns; Xte = enc(test, cols)
        res = {}
        for name, clf in [("GBM", HistGradientBoostingClassifier(random_state=SEED)),
                          ("LogReg", LogisticRegression(max_iter=2000))]:
            clf.fit(Xtr, ytr); pp = clf.predict_proba(Xte)[:, 1]; pr = (pp >= 0.5).astype(int)
            res[name] = {"acc": round(accuracy_score(yte, pr), 4), "f1": round(f1_score(yte, pr, average="macro"), 4),
                         "auc": round(roc_auc_score(yte, pp), 4), "ap": round(average_precision_score(yte, pp), 4)}
        return res

    # ===== ABLATION 1: feature groups =====
    print("[ablation] feature groups...")
    static = [c for c in ALL if c not in COLLAB]
    out["ablation_features"] = {
        "static_only": fit_eval(static),
        "+user&streamer_ctr": fit_eval(static + ["user_ctr", "streamer_ctr"]),
        "full": fit_eval(ALL)}
    for k, v in out["ablation_features"].items():
        print(f"   {k:<20} GBM acc {v['GBM']['acc']}  LogReg acc {v['LogReg']['acc']}")

    # ---------- LLM batch predictor (hard pred + P(click) via chosen-token logprob) ----------
    def llm_run(rows, temp=0.0, order=None):
        feats_backup = P.FEATURES
        if order is not None:
            P.FEATURES = order
        hp = np.zeros(len(rows), dtype=int); pp = np.full(len(rows), 0.5)
        def one(i):
            msg = [{"role": "user", "content": PROMPT_PREFIX + P.serialize(rows.iloc[i]) + QUESTION}]
            for a in range(4):
                try:
                    r = client.chat.completions.create(model=FT_PATH, messages=msg, max_tokens=1,
                            temperature=temp, logprobs=True, top_logprobs=0)
                    ct = r.choices[0].logprobs.content[0]; tok = ct.token.strip(); pe = math.exp(ct.logprob)
                    prob = pe if tok == "1" else (1 - pe) if tok == "0" else 0.5
                    return i, int(prob >= 0.5), prob
                except Exception:
                    time.sleep(2 ** a)
            return i, 0, 0.5
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
                i, h, pr = fut.result(); hp[i] = h; pp[i] = pr
        P.FEATURES = feats_backup
        return hp, pp

    def m(y, h, p):
        return {"acc": round(accuracy_score(y, h), 4), "f1": round(f1_score(y, h, average="macro"), 4),
                "bal_acc": round(balanced_accuracy_score(y, h), 4),
                "auc": round(roc_auc_score(y, p), 4), "ap": round(average_precision_score(y, p), 4)}

    # ===== canonical (T=0) on balanced test: hard preds + probs =====
    print("[llm] canonical T=0 ...")
    h0, p0 = llm_run(test, temp=0.0)
    out["llm_canonical"] = m(yte, h0, p0)
    print("   ", out["llm_canonical"])

    # ===== ROBUSTNESS 2: temperature =====
    print("[robust] temperature T=1 ...")
    h1, p1 = llm_run(test, temp=1.0)
    out["robust_temperature"] = {"T0": out["llm_canonical"], "T1": m(yte, h1, p1),
                                 "agreement_T0_T1": round(float((h0 == h1).mean()), 4)}
    print("   ", out["robust_temperature"])

    # ===== ROBUSTNESS 3: feature-order permutation =====
    print("[robust] feature-order shuffle ...")
    rng = np.random.RandomState(SEED); shuf = ALL.copy(); rng.shuffle(shuf)
    hs, ps = llm_run(test, temp=0.0, order=shuf)
    out["robust_feature_order"] = {"shuffled": m(yte, hs, ps),
                                   "agreement_with_canonical": round(float((h0 == hs).mean()), 4)}
    print("   ", out["robust_feature_order"])

    # ===== ROBUSTNESS 4: prior shift (natural ~28% CTR test) =====
    print("[robust] natural prior shift ...")
    npos = int(round(NAT_RATE * NAT_N)); nneg = NAT_N - npos
    pos = pool[pool["clicked"] == 1].head(npos); neg = pool[pool["clicked"] == 0].head(nneg)
    nat = pd.concat([pos, neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    yn = nat["clicked"].values
    hn, pn = llm_run(nat, temp=0.0)
    out["robust_prior_shift"] = {"rate": round(float(yn.mean()), 3), "n": int(len(nat)), **m(yn, hn, pn)}
    print("   ", out["robust_prior_shift"])

    # ===== ABLATION/Discrimination summary: AUC vs baselines (balanced) =====
    out["discrimination_balanced"] = {"RashomonLLM": {k: out["llm_canonical"][k] for k in ("auc", "ap", "acc")},
                                      "GBM": {k: out["ablation_features"]["full"]["GBM"][k] for k in ("auc", "ap", "acc")},
                                      "LogReg": {k: out["ablation_features"]["full"]["LogReg"][k] for k in ("auc", "ap", "acc")}}

    json.dump(out, open(os.path.join(HERE, "analysis", "validity_result.json"), "w", encoding="utf-8"), indent=2)
    print("\nsaved analysis/validity_result.json")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
