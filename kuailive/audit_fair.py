"""Subgroup fairness audit of the deployed v2 predict-only RashomonLLM model.
Scores a held-out test window (rows 30000:32500, disjoint from training) and
reports accuracy / FPR / FNR by user gender and age band."""
import os
os.environ["FT_DATA"] = r"D:\Big_LLM_Project\KuaiLive\analysis\ctr_dataset_v2.csv"
os.environ["FT_TRAIN"] = "30000"
os.environ["FT_TEST"] = "2500"
import numpy as np, re, time
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
            r = client.chat.completions.create(model=path, messages=msg, max_tokens=8, temperature=0.0)
            txt = r.choices[0].message.content or ""
            m = re.findall(r"[01]", txt)
            return i, (int(m[0]) if m else 0)
        except Exception:
            time.sleep(2 ** a)
    return i, 0

preds = [None] * len(rows)
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
        i, p = fut.result(); preds[i] = p
preds = np.array([0 if p is None else p for p in preds])
y = rows[P.TARGET].values
g = rows["u_gender"].astype(str).values
age = rows["u_age"].astype(str).values

def rates(mask):
    yy, pp = y[mask], preds[mask]
    n = len(yy)
    acc = round((yy == pp).mean(), 3) if n else float("nan")
    fpr = round(((pp == 1) & (yy == 0)).sum() / max((yy == 0).sum(), 1), 3)
    fnr = round(((pp == 0) & (yy == 1)).sum() / max((yy == 1).sum(), 1), 3)
    return f"n={n} acc={acc} fpr={fpr} fnr={fnr}"

print("AUDIT_RESULT OVERALL", rates(np.ones(len(y), bool)))
for gv in sorted(set(g)):
    print("AUDIT_RESULT GENDER", gv, rates(g == gv))
for av in ["12-17", "18-23", "24-30", "31-40", "41-49", "50+"]:
    print("AUDIT_RESULT AGE", av, rates(age == av))
print("AUDIT_DONE elapsed", round(time.time() - t0), "s")
