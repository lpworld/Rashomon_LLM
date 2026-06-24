# -*- coding: utf-8 -*-
"""
Generate concrete RashomonLLM explanation EXAMPLES on real KuaiLive v2 test instances.

For a handful of illustrative test exposures spanning outcome types, we report:
  - the key feature values (incl. time-aware collaborative CTRs),
  - the fine-tuned RashomonLLM hard prediction (the 0.771 v2 model),
  - a one-sentence natural-language rationale RashomonLLM emits for that prediction.

Prediction comes from the fine-tuned predictor; the rationale is articulated by the
same backbone conditioned on the features and its own prediction (the self-explaining
behavior of the EPR Explanation Agent). Runs on Tinker credit.
"""
import os, sys, re, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P

DATA = os.path.join(HERE, "analysis", "ctr_dataset_v2.csv")
TINKER = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
FT_PATH = open(os.path.join(HERE, "analysis", "ft_sampler_path.txt"), encoding="utf-8").read().strip()
BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FT_TRAIN, FT_TEST, SEED = 30000, 2000, 42

PROMPT_PREFIX = ("You are predicting user engagement on a live-streaming platform.\n"
                 "Exposure features:\n")
QUESTION = ("\n\nWill this user click into this live room? "
            "Answer with a single digit: 1 = click, 0 = no click.")
EXPL_SYS = ("You are RashomonLLM, an interpretable click-through predictor for a live-streaming "
            "platform. In ONE concise sentence (<= 40 words), explain WHY the model predicts the "
            "given outcome, citing the THREE most decision-relevant feature VALUES. Prefer the "
            "historical click-through rates (user_ctr, streamer_ctr, cat_ctr), user activity, "
            "streamer popularity, and time-of-day. IMPORTANT REFERENCE: across all exposures the "
            "average click-through rate is about 0.28, so treat a CTR well below 0.28 as LOW and "
            "well above 0.28 as HIGH; be consistent about this. State it as a clear reason; do not hedge.")

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
    client = OpenAI(base_url=TINKER, api_key=os.environ["TINKER_API_KEY"])
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    P.FEATURES = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]
    test = df.iloc[FT_TRAIN:FT_TRAIN + FT_TEST].reset_index(drop=True)

    uc, sc, cc = test["user_ctr"], test["streamer_ctr"], test["cat_ctr"]
    uq = uc.quantile([.25, .75]); sq = sc.quantile([.25, .75]); un = test["user_n"]
    def first(mask, label):
        sub = test[(mask) & (test["clicked"] == label)]
        return None if len(sub) == 0 else sub.index[0]
    picks = {
        "A. High-affinity click (user & streamer both hot)":
            first((uc >= uq[.75]) & (sc >= sq[.75]), 1),
        "B. Cold exposure, no click (both CTRs low)":
            first((uc <= uq[.25]) & (sc <= sq[.25]), 0),
        "C. Streamer-pull click (low user_ctr, high streamer_ctr)":
            first((uc <= uq[.25]) & (sc >= sq[.75]), 1),
        "D. Active user skips a weak streamer (high user_n, low streamer_ctr)":
            first((un >= un.quantile(.75)) & (sc <= sq[.25]), 0),
        "E. Category-driven click (high cat_ctr)":
            first(cc >= cc.quantile(.80), 1),
    }

    def ft_predict(row):
        msg = [{"role": "user", "content": PROMPT_PREFIX + P.serialize(row) + QUESTION}]
        r = client.chat.completions.create(model=FT_PATH, messages=msg, max_tokens=1, temperature=0)
        m = re.search(r"[01]", r.choices[0].message.content or ""); return int(m.group()) if m else 0

    def explain(row, decision):
        u = ("Exposure features:\n" + P.serialize(row) +
             f"\n\nThe model predicts: {decision}. Explain in one sentence starting with 'Because'.")
        r = client.chat.completions.create(model=BASE, max_tokens=160, temperature=0.0,
            messages=[{"role": "system", "content": EXPL_SYS}, {"role": "user", "content": u}])
        return (r.choices[0].message.content or "").strip().replace("\n", " ")

    KEY = ["user_ctr", "user_n", "streamer_ctr", "streamer_n", "cat_ctr",
           "r_content_category", "ctx_hour", "u_accu_watch_live_cnt", "s_fans_user_num"]
    out = []
    for title, idx in picks.items():
        if idx is None:
            print(f"\n### {title}\n  (no matching instance)"); continue
        row = test.loc[idx]
        pred = ft_predict(row)
        decision = "click (1)" if pred == 1 else "no click (0)"
        rat = explain(row, decision)
        kv = {k: (round(float(row[k]), 4) if pd.api.types.is_number(row[k]) else str(row[k]))
              for k in KEY if k in row}
        rec = {"case": title, "true": int(row["clicked"]), "pred": pred,
               "key_features": kv, "rationale": rat}
        out.append(rec)
        print(f"\n### {title}")
        print(f"  key: {kv}")
        print(f"  true={int(row['clicked'])}  pred={pred}  ({'correct' if pred==int(row['clicked']) else 'WRONG'})")
        print(f"  rationale: {rat.encode('ascii','replace').decode()}")
    json.dump(out, open(os.path.join(HERE, "analysis", "explanation_examples.json"), "w"), indent=2)
    print("\nsaved analysis/explanation_examples.json")

if __name__ == "__main__":
    main()
