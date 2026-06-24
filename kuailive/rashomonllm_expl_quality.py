# -*- coding: utf-8 -*-
"""
Explanation-quality evaluation for RashomonLLM on KuaiLive CTR (the headline contribution).

Two Nauta-style, model-agnostic metrics, computed on a strong common reference predictor
(GBM) so different explanations are compared on equal footing:

  - SINGLE-DELETION (faithfulness): ablate the top-k features an explanation ranks as
    important (mean/mode-impute) and measure the predictor's accuracy DROP. A faithful
    explanation flags features the model truly relies on -> bigger drop is better.
  - STABILITY / RANDOMIZATION: perturb the features an explanation ranks as UNimportant
    (bottom-k) and measure how often predictions stay the same. A good explanation
    correctly identifies non-influential features -> higher stability is better.

Compared rankings:
  RashomonLLM (LLM-reasoned global importance) vs Sparse Decision Tree (XAI baseline)
  vs Permutation Importance (gold reference) vs Random (floor).
Only the RashomonLLM ranking needs the LLM (one reasoned call); everything else is local.
"""
import os, sys, re, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P

DATA = os.path.join(HERE, "analysis", "ctr_dataset_v2.csv")
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FT_TRAIN, FT_TEST, SEED = 30000, 2000, 42

def load_tinker_key():
    for line in open(os.path.join(HERE, "tinker_key.local"), encoding="utf-8"):
        s = line.strip()
        if s and not s.startswith("#"):
            if "=" in s and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s.split("=", 1)[0].strip()):
                s = s.split("=", 1)[1]
            os.environ["TINKER_API_KEY"] = s.strip().strip('"').strip("'")
            return

FEATURE_GLOSS = {
    "user_ctr": "user's historical click-through rate", "user_n": "user's past exposure count",
    "streamer_ctr": "streamer's historical CTR", "streamer_n": "streamer's past exposure count",
    "cat_ctr": "content-category historical CTR",
}

def main():
    load_tinker_key()
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    FEATS = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]
    P.FEATURES = FEATS
    train = df.iloc[:FT_TRAIN].reset_index(drop=True)
    test = df.iloc[FT_TRAIN:FT_TRAIN + FT_TEST].reset_index(drop=True)
    ytr, yte = train["clicked"].values, test["clicked"].values

    # --- reference predictor (GBM) with consistent encoding ---
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    num = [c for c in FEATS if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() > 30]
    cat = [c for c in FEATS if c not in num]
    def encode(frame):
        X = pd.concat([pd.get_dummies(frame[cat].astype(str)),
                       frame[num].reset_index(drop=True)], axis=1)
        return X.reindex(columns=COLS, fill_value=0)
    Xtr = pd.concat([pd.get_dummies(train[cat].astype(str)), train[num].reset_index(drop=True)], axis=1)
    COLS = Xtr.columns
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    Xte = encode(test)
    base_pred = gbm.predict(Xte)
    base_acc = accuracy_score(yte, base_pred)
    print(f"reference GBM test acc: {base_acc:.4f}  | {len(FEATS)} features")

    # map each original feature -> its encoded columns (for ablation we work on original frame)
    def ablate(frame, feats_to_kill):
        t = frame.copy()
        for f in feats_to_kill:
            t[f] = train[f].mean() if f in num else train[f].mode().iloc[0]
        return t

    # --- rankings ---
    rankings = {}

    # gold: permutation importance on GBM (aggregate encoded importance back to original feature)
    print("computing permutation importance (gold)...")
    pi = permutation_importance(gbm, Xte, yte, n_repeats=5, random_state=SEED, n_jobs=-1)
    col_imp = dict(zip(COLS, pi.importances_mean))
    feat_imp = {}
    for f in FEATS:
        if f in num:
            feat_imp[f] = col_imp.get(f, 0.0)
        else:
            feat_imp[f] = sum(v for c, v in col_imp.items() if c.startswith(f + "_"))
    rankings["Permutation(gold)"] = sorted(FEATS, key=lambda f: -feat_imp[f])

    # XAI baseline: sparse (shallow) decision tree
    dt = DecisionTreeClassifier(max_depth=6, random_state=SEED).fit(Xtr, ytr)
    dt_col = dict(zip(COLS, dt.feature_importances_))
    dt_imp = {f: (dt_col.get(f, 0.0) if f in num else sum(v for c, v in dt_col.items() if c.startswith(f + "_"))) for f in FEATS}
    rankings["SparseTree(XAI)"] = sorted(FEATS, key=lambda f: -dt_imp[f])

    # random floor
    rng = np.random.RandomState(SEED); rnd = FEATS.copy(); rng.shuffle(rnd)
    rankings["Random"] = rnd

    # RashomonLLM: LLM-reasoned global ranking
    print("querying RashomonLLM for feature ranking...")
    from openai import OpenAI
    client = OpenAI(base_url=TINKER_BASE_URL, api_key=os.environ["TINKER_API_KEY"])
    feats_desc = "\n".join(f"- {f}" + (f" ({FEATURE_GLOSS[f]})" if f in FEATURE_GLOSS else "") for f in FEATS)
    prompt = ("You are an expert on live-streaming recommendation. Below are features describing a "
              "user-streamer exposure. Rank ALL of them from MOST to LEAST important for predicting "
              "whether the user clicks into the live room. Output ONLY the feature names, one per line, "
              "most important first, using the exact names given.\n\n" + feats_desc)
    txt = client.chat.completions.create(model=BASE_MODEL, max_tokens=700, temperature=0.0,
            messages=[{"role": "user", "content": prompt}]).choices[0].message.content or ""
    pos = {}
    for f in FEATS:
        m = re.search(re.escape(f), txt)
        pos[f] = m.start() if m else 10**9
    rankings["RashomonLLM"] = sorted(FEATS, key=lambda f: pos[f])

    # --- metrics ---
    def acc_after_delete(ranking, k):
        return accuracy_score(yte, gbm.predict(encode(ablate(test, ranking[:k]))))
    def stability_perturb_bottom(ranking, k):
        kill = ranking[-k:]
        p = gbm.predict(encode(ablate(test, kill)))
        return (p == base_pred).mean()

    print("\n================= EXPLANATION QUALITY =================")
    print(f"(reference GBM acc = {base_acc:.4f}; deletion -> lower acc = MORE faithful; stability -> higher = better)\n")
    rows = []
    for name, rk in rankings.items():
        a1, a3, a5 = acc_after_delete(rk, 1), acc_after_delete(rk, 3), acc_after_delete(rk, 5)
        stab = stability_perturb_bottom(rk, 10)
        rows.append((name, rk[0], a1, a3, a5, base_acc - a5, stab))
    print(f"{'method':<18} {'top-1 feat':<16} {'del1':>6} {'del3':>6} {'del5':>6} {'drop@5':>7} {'stab':>6}")
    for name, top, a1, a3, a5, drop, stab in rows:
        print(f"{name:<18} {top:<16} {a1:>6.4f} {a3:>6.4f} {a5:>6.4f} {drop:>7.4f} {stab:>6.4f}")

    rec = {"base_acc": base_acc,
           "rankings": {k: v[:8] for k, v in rankings.items()},
           "results": [{"method": n, "top1": t, "del1": a1, "del3": a3, "del5": a5,
                        "drop_at_5": d, "stability": s} for n, t, a1, a3, a5, d, s in rows]}
    json.dump(rec, open(os.path.join(HERE, "analysis", "expl_quality_result.json"), "w"), indent=2)
    print("\nsaved expl_quality_result.json")
    print("\nRashomonLLM top-8:", rankings["RashomonLLM"][:8])
    print("Gold        top-8:", rankings["Permutation(gold)"][:8])

if __name__ == "__main__":
    main()
