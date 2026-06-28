"""Equalized-odds audit + equal-opportunity threshold recalibration of the
deployed v2 predict-only RashomonLLM model (same model/window as audit_fair.py).

Scores the held-out window (rows 30000:32500) producing a calibrated P(click)
via the chosen-token logprob, then reports per-subgroup FPR/FNR (1) at the
global 0.5 threshold and (2) after per-group threshold recalibration that
equalizes true-positive rate (equal opportunity; Hardt, Price & Srebro 2016).
Prints subgroup sizes, per-group rates, max-min spreads, and the accuracy cost."""
import os, math, time
os.environ["FT_DATA"]  = r"D:\Big_LLM_Project\KuaiLive\analysis\ctr_dataset_v2.csv"
os.environ["FT_TRAIN"] = "30000"
os.environ["FT_TEST"]  = "2500"
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import rashomonllm_finetune as M
import rashomonllm_ft_epr as F          # reuse oai() Tinker client
import rashomonllm_pilot as P

M.load_tinker_key()
path = open(r"analysis/ft_sampler_path.txt").read().strip()
print("model:", path)
train_df, test_df = M.load_split()
rows = test_df.reset_index(drop=True)
client = F.oai()

def one(i):
    msg = [{"role": "user", "content": M.to_user_prompt(rows.iloc[i])}]
    for a in range(4):
        try:
            r = client.chat.completions.create(model=path, messages=msg, max_tokens=1,
                    temperature=0.0, logprobs=True, top_logprobs=0)
            ct = r.choices[0].logprobs.content[0]
            tok = ct.token.strip(); pe = math.exp(ct.logprob)
            prob = pe if tok == "1" else (1 - pe) if tok == "0" else 0.5
            return i, float(prob)
        except Exception:
            time.sleep(2 ** a)
    return i, 0.5

prob = np.full(len(rows), 0.5)
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
        i, pr = fut.result(); prob[i] = pr
print("scored", len(rows), "rows in", round(time.time() - t0), "s")

y   = rows[P.TARGET].values.astype(int)
gen = rows["u_gender"].astype(str).values
age = rows["u_age"].astype(str).values

def rates(yy, pp):
    n = len(yy)
    acc = (yy == pp).mean() if n else float("nan")
    npos = max((yy == 1).sum(), 1); nneg = max((yy == 0).sum(), 1)
    fpr = ((pp == 1) & (yy == 0)).sum() / nneg
    fnr = ((pp == 0) & (yy == 1)).sum() / npos
    return n, acc, fpr, fnr

def report(tag, groups, labels, pred):
    print(f"\n=== {tag} ===")
    n, acc, fpr, fnr = rates(y, pred)
    print(f"OVERALL n={n} acc={acc:.3f} fpr={fpr:.3f} fnr={fnr:.3f}")
    fprs, fnrs = [], []
    for lab in labels:
        m = groups == lab
        if m.sum() == 0: continue
        gn, ga, gf, gn2 = rates(y[m], pred[m])
        print(f"  {lab:<7} n={gn:<5} acc={ga:.3f} fpr={gf:.3f} fnr={gn2:.3f}")
        fprs.append(gf); fnrs.append(gn2)
    if fprs:
        print(f"  SPREAD fpr={max(fprs)-min(fprs):.3f}  fnr={max(fnrs)-min(fnrs):.3f}")
    return acc

# ---- baseline: global 0.5 threshold ----
pred0 = (prob >= 0.5).astype(int)
GEN = sorted(set(gen))
AGE = ["12-17", "18-23", "24-30", "31-40", "41-49", "50+"]
# coarse age bands for stability
def coarse(a):
    return {"12-17":"<24","18-23":"<24","24-30":"24-40","31-40":"24-40",
            "41-49":"40+","50+":"40+"}.get(a, a)
agec = np.array([coarse(a) for a in age]); AGEC = ["<24","24-40","40+"]

print("\n########## BASELINE (global tau=0.5) ##########")
report("Gender", gen, GEN, pred0)
report("Age (fine)", age, AGE, pred0)
report("Age (coarse)", agec, AGEC, pred0)

# ---- equal-opportunity recalibration: per-group threshold to match overall TPR ----
overall_tpr = 1.0 - rates(y, pred0)[3]    # global TPR at tau=0.5
def recalibrate(groups, labels):
    pred = pred0.copy()
    th = {}
    for lab in labels:
        m = groups == lab
        pos = prob[m & (y == 1)]
        tau = float(np.quantile(pos, 1 - overall_tpr)) if len(pos) else 0.5
        th[lab] = tau
        pred[m] = (prob[m] >= tau).astype(int)
    return pred, th

print(f"\n########## RECALIBRATED (equal-opportunity, target TPR={overall_tpr:.3f}) ##########")
for tag, groups, labels in [("Gender", gen, GEN), ("Age (fine)", age, AGE), ("Age (coarse)", agec, AGEC)]:
    pred, th = recalibrate(groups, labels)
    acc = report(tag, groups, labels, pred)
    print("  thresholds:", {k: round(v, 3) for k, v in th.items()})
print("\nAUDIT_DONE elapsed", round(time.time() - t0), "s")
