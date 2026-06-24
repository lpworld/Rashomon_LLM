# -*- coding: utf-8 -*-
"""
#3 EPR correction on the v2 fine-tuned model (KuaiLive CTR).

Faithful test of the Reflection mechanism on a strong, saturated predictor:
  1. Run v2 FT model on a held-out validation slice; collect its mistakes.
  2. Reflection: a base Qwen reads the systematic errors and writes a correction guide.
  3. Re-predict the test set with that guide prepended; compare FT-alone vs FT+reflection.

Expectation: limited gain (FT errors ~irreducible & GBM-correlated), but this directly
tests whether reflection lifts a saturated predictor. Runs on Tinker credit.
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
FT_PATH = open(os.path.join(HERE, "analysis", "ft_sampler_path.txt"), encoding="utf-8").read().strip()
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FT_TRAIN, FT_TEST, VAL_N, SEED, CONC = 30000, 2000, 1000, 42, 8
N_ERR_FEED = 50

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

def score(y, pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": round(accuracy_score(y, pred), 4), "f1": round(f1_score(y, pred, average="macro"), 4)}

def main():
    load_tinker_key()
    from openai import OpenAI
    client = OpenAI(base_url=TINKER_BASE_URL, api_key=os.environ["TINKER_API_KEY"])

    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    P.FEATURES = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]
    test = df.iloc[FT_TRAIN:FT_TRAIN + FT_TEST].reset_index(drop=True)
    val  = df.iloc[FT_TRAIN + FT_TEST:FT_TRAIN + FT_TEST + VAL_N].reset_index(drop=True)
    print(f"FT model: {FT_PATH}\ntest={len(test)} val={len(val)} feats={len(P.FEATURES)}")

    def ft_predict(rows, guide=None):
        preds = [0] * len(rows)
        prefix = (f"CORRECTION NOTES (from analyzing prediction errors):\n{guide}\n\n" if guide else "")
        def one(i):
            msg = [{"role": "user", "content": prefix + PROMPT_PREFIX + P.serialize(rows.iloc[i]) + QUESTION}]
            for a in range(4):
                try:
                    r = client.chat.completions.create(model=FT_PATH, messages=msg, max_tokens=1, temperature=0)
                    m = re.search(r"[01]", r.choices[0].message.content or "")
                    return i, (int(m.group()) if m else 0)
                except Exception:
                    time.sleep(2 ** a)
            return i, 0
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
                i, p = fut.result(); preds[i] = p
        return np.array(preds)

    # 1. FT on validation -> mistakes
    print("[1] FT predicting validation...")
    yv = val["clicked"].values
    pv = ft_predict(val)
    print("    val FT:", score(yv, pv))
    wrong = val[pv != yv].copy(); wrong["_pred"] = pv[pv != yv]
    wrong = wrong.head(N_ERR_FEED)

    # 2. Reflection -> correction guide
    print("[2] Reflection: writing correction guide from", len(wrong), "errors...")
    err_txt = "\n".join(f"[true={int(r['clicked'])}, predicted={int(r['_pred'])}] {P.serialize(r)}"
                        for _, r in wrong.iterrows())
    refl = client.chat.completions.create(model=BASE_MODEL, max_tokens=400, temperature=0.2,
        messages=[{"role": "system", "content":
                   "You analyze a click-predictor's systematic errors and write concise, actionable "
                   "correction guidance (<=180 words): in which feature regimes does it over- or "
                   "under-predict clicks, and how should it adjust."},
                  {"role": "user", "content": f"Mispredicted exposures (true vs predicted):\n{err_txt}"}])
    guide = (refl.choices[0].message.content or "").strip()
    print("    guide:", guide[:280].replace("\n", " ").encode("ascii", "replace").decode(), "...")

    # 3. Re-predict test with and without the guide
    yt = test["clicked"].values
    print("[3] FT on test (no guide)...")
    p_base = ft_predict(test)
    print("    FT-alone:", score(yt, p_base))
    print("[3] FT on test (+ reflection guide)...")
    p_corr = ft_predict(test, guide=guide)
    print("    FT+EPR  :", score(yt, p_corr))

    print("\n============= EPR-CORRECTION RESULT =============")
    print(f"  GBM (ref)   : acc 0.7525")
    print(f"  FT-alone    : {score(yt, p_base)}")
    print(f"  FT + EPR    : {score(yt, p_corr)}")
    changed = int((p_base != p_corr).sum())
    print(f"  predictions changed by guide: {changed}/{len(test)}")
    rec = {"ft_path": FT_PATH, "ft_alone": score(yt, p_base), "ft_epr": score(yt, p_corr),
           "val_ft": score(yv, pv), "changed": changed, "guide": guide}
    json.dump(rec, open(os.path.join(HERE, "analysis", "epr_correct_result.json"), "w"), indent=2)
    print("saved epr_correct_result.json")

if __name__ == "__main__":
    main()
