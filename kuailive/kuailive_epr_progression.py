# -*- coding: utf-8 -*-
"""
Capture the PROGRESSION of the RashomonLLM explanation across EPR (Explanation ->
Prediction -> Reflection) iterations on KuaiLive, together with the validation accuracy
at each step. Shows how double-loop reflection rewrites the click-pattern description and
how accuracy improves from the one-shot explanation to the EPR-refined one.

Reuses the pilot's agents. Runs on Tinker credit with a stable instruct backbone.
Saves analysis/epr_progression.json (full explanation text per iteration + accuracies).
"""
import os, sys, json
os.environ.setdefault("PILOT_PROVIDER", "tinker")
os.environ.setdefault("PILOT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
os.environ.setdefault("PILOT_LEARN", "300")
os.environ.setdefault("PILOT_VALID", "300")
os.environ.setdefault("PILOT_TEST", "1000")
os.environ.setdefault("PILOT_ITERS", "3")
os.environ.setdefault("PILOT_EXPLROWS", "70")
os.environ.setdefault("PILOT_PREDBATCH", "50")
os.environ.setdefault("PILOT_MAX_USD", "999")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import rashomonllm_pilot as P

# Use the v2 collaborative-feature dataset (the one the KuaiLive section is built on),
# where in-context prediction has real signal for reflection to refine.
P.DATA = os.path.join(HERE, "analysis", "ctr_dataset_v2.csv")
_df = pd.read_csv(P.DATA)
P.FEATURES = [c for c in _df.columns if c not in ("user_id", "live_id", "streamer_id", "clicked")]

def excerpt(t, n=400):
    t = " ".join(t.split())
    return (t[:n] + " ...") if len(t) > n else t

def main():
    P._load_key()
    full, learn, valid, test = P.load_splits()
    yv, yt = valid[P.TARGET].values, test[P.TARGET].values
    print(f"learn={len(learn)} valid={len(valid)} test={len(test)} model={P.MODEL}")

    steps = []  # (label, val_acc, explanation_text)
    E = P.explanation_agent(learn)
    print("\n[one-shot] scoring initial explanation on test...")
    p0, _ = P.predict_batch(E, test); oneshot = P.score(yt, p0)
    print("   one-shot test:", oneshot)

    cur = E
    for it in range(0, 3):
        pv, _ = P.predict_batch(cur, valid); sv = P.score(yv, pv)
        label = "Iteration 0 (initial explanation)" if it == 0 else f"Iteration {it} (after {it} reflection step{'s' if it>1 else ''})"
        steps.append((label, sv, cur))
        print(f"\n=== {label} ===  valid acc={sv['acc']} f1={sv['f1_macro']}")
        print("   expl:", excerpt(cur).encode("ascii", "replace").decode())
        cur = P.reflection_agent(cur, valid, pv)

    # final refined explanation evaluated on validation + test
    pvf, _ = P.predict_batch(cur, valid); svf = P.score(yv, pvf)
    steps.append(("Iteration 3 (EPR-refined explanation)", svf, cur))
    print(f"\n=== Iteration 3 (EPR-refined) ===  valid acc={svf['acc']} f1={svf['f1_macro']}")
    print("   expl:", excerpt(cur).encode("ascii", "replace").decode())
    pf, _ = P.predict_batch(cur, test); final = P.score(yt, pf)
    print("\n[final] EPR-refined test:", final)
    print(f"\nEPR lift (test acc): {oneshot['acc']} -> {final['acc']} "
          f"(+{round(final['acc']-oneshot['acc'],4)})")

    rec = {"model": P.MODEL, "oneshot_test": oneshot, "epr_final_test": final,
           "steps": [{"label": l, "valid": s, "explanation": e} for l, s, e in steps]}
    json.dump(rec, open(os.path.join(HERE, "analysis", "epr_progression.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nsaved analysis/epr_progression.json")

if __name__ == "__main__":
    main()
