# -*- coding: utf-8 -*-
"""
RashomonLLM full method on KuaiLive CTR: Data-Enhanced (fine-tuned) LLM that is ALSO
self-explaining (EPR). Instead of a constant global explanation (which a fine-tuned
predictor learns to ignore), the explanation is PER-EXAMPLE and part of the model's
own output: the model emits a short feature-grounded rationale, then the label.

Pipeline:
  1. TEACHER  : base Qwen writes a one-sentence, non-leaky, feature-grounded rationale
                for each training row (does NOT state the outcome).
  2. FINETUNE : LoRA SFT on  features -> "<rationale>\nAnswer: <0/1>"  (CoT target).
  3. EVAL     : fine-tuned model self-explains then answers; parse the answer.
                Compare to LogReg/GBM baselines and to plain-FT (features->label).

Runs on Tinker (TML) credit. Key from tinker_key.local / TINKER_API_KEY.
Resumable: FT_EPR_STAGE = teacher | finetune | eval  (artifacts cached to analysis/).
"""
import os, sys, re, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P   # serialize(), FEATURES, TARGET, score()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
DATA = os.path.join(OUT, "ctr_dataset.csv")
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
RATIONALE_FILE = os.path.join(OUT, "ft_epr_rationales.jsonl")
PATH_FILE = os.path.join(OUT, "ft_epr_sampler_path.txt")

def _envi(n, d): return int(os.environ.get(n, d))
BASE_MODEL = os.environ.get("FT_EPR_BASE", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TRAIN_N = _envi("FT_EPR_TRAIN", 3000)
TEST_N  = _envi("FT_EPR_TEST", 1000)
EPOCHS  = _envi("FT_EPR_EPOCHS", 2)
BATCH   = _envi("FT_EPR_BATCH", 64)
LR      = float(os.environ.get("FT_EPR_LR", "1e-4"))
RANK    = _envi("FT_EPR_RANK", 32)
CONC    = _envi("FT_EPR_CONC", 8)
RAT_W   = float(os.environ.get("FT_EPR_RAT_W", "0.05"))  # loss weight on rationale tokens (answer=1.0)
MAXLEN  = 2048
SEED    = 42
STAGE   = os.environ.get("FT_EPR_STAGE", "all")  # all | teacher | finetune | eval

# ----------------------------- key ----------------------------------
def load_tinker_key():
    if os.environ.get("TINKER_API_KEY"):
        return
    kf = os.path.join(HERE, "tinker_key.local")
    with open(kf, encoding="utf-8") as f:
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
                os.environ["TINKER_API_KEY"] = line
                return
    sys.exit("ERROR: no TINKER_API_KEY (env or tinker_key.local).")

def oai():
    from openai import OpenAI
    return OpenAI(base_url=TINKER_BASE_URL, api_key=os.environ["TINKER_API_KEY"])

# ----------------------------- data ---------------------------------
def load_split():
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train = df.iloc[:TRAIN_N].reset_index(drop=True)
    test  = df.iloc[TRAIN_N:TRAIN_N + TEST_N].reset_index(drop=True)
    return train, test

def baselines(train, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    comb = pd.concat([train, test], ignore_index=True)
    X = pd.get_dummies(comb[P.FEATURES].astype(str))
    Xtr, Xte = X.iloc[:len(train)], X.iloc[len(train):]
    ytr, yte = train[P.TARGET].values, test[P.TARGET].values
    return {"LogReg": P.score(yte, LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict(Xte)),
            "GBM":    P.score(yte, HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr).predict(Xte))}

# --------------------------- teacher --------------------------------
TEACHER_SYS = ("You are an analyst of live-streaming engagement. Given a user-streamer "
               "exposure, write ONE concise sentence (<= 30 words) describing the "
               "engagement-relevant signal in the features (e.g., content niche fit, the "
               "user's activity level, streamer popularity, time-of-day). Cite specific "
               "feature values. Do NOT state or guess whether the user clicks.")

def gen_rationales(train_df):
    client = oai()
    rows = train_df.reset_index(drop=True)
    out = [None] * len(rows)
    def one(i):
        u = "Exposure features:\n" + P.serialize(rows.iloc[i]) + "\n\nOne-sentence signal:"
        for a in range(4):
            try:
                r = client.chat.completions.create(model=BASE_MODEL,
                        messages=[{"role": "system", "content": TEACHER_SYS},
                                  {"role": "user", "content": u}],
                        max_tokens=80, temperature=0.3)
                return i, (r.choices[0].message.content or "").strip().replace("\n", " ")
            except Exception:
                time.sleep(2 ** a)
        return i, "Mixed engagement signals across user activity, streamer niche, and timing."
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
            i, txt = fut.result(); out[i] = txt; done += 1
            if done % 200 == 0:
                print(f"[teacher] {done}/{len(rows)} ({time.time()-t0:.0f}s)")
    with open(RATIONALE_FILE, "w", encoding="utf-8") as f:
        for i in range(len(rows)):
            f.write(json.dumps({"rationale": out[i], "label": int(rows.iloc[i][P.TARGET])}) + "\n")
    print(f"[teacher] wrote {len(rows)} rationales -> {RATIONALE_FILE}")
    return out

# prompt the model sees (train + inference identical) -- ANSWER FIRST, then explain
PRED_USER = ("Exposure features:\n{feat}\n\nFirst give the click prediction as "
             "`Answer: 1` (click) or `Answer: 0` (no click). Then on a new line explain "
             "in one sentence, citing the key features (`Because: ...`).")
def pred_user(row): return PRED_USER.format(feat=P.serialize(row))
def train_assistant(rationale, label): return f"Answer: {label}\nBecause: {rationale}"

# --------------------------- finetune -------------------------------
def finetune(train_df, rationales):
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import datum_from_model_input_weights
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    tok = get_tokenizer(BASE_MODEL)
    rname = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(rname, tok)
    rows = train_df.reset_index(drop=True)
    labels = [int(rows.iloc[i][P.TARGET]) for i in range(len(rows))]
    msgs = [[{"role": "user", "content": pred_user(rows.iloc[i])},
             {"role": "assistant", "content": train_assistant(rationales[i], labels[i])}]
            for i in range(len(rows))]
    print(f"[ft] model={BASE_MODEL} renderer={rname} rows={len(msgs)} batch={BATCH} "
          f"epochs={EPOCHS} rationale_w={RAT_W}")

    def wdatum(i):
        # Weight loss toward the answer: keep full weight on the leading "Answer: X"
        # tokens, downweight the trailing rationale so prediction dominates (like plain-FT).
        mi, w = renderer.build_supervised_example(
            msgs[i], train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES)
        w = np.asarray(w, dtype=float).copy()
        k = len(tok.encode(f"Answer: {labels[i]}\n"))   # answer-region token count (approx)
        nz = np.nonzero(w > 0)[0]
        if len(nz) > k:
            w[nz[k:]] *= RAT_W
        return datum_from_model_input_weights(mi, w, MAXLEN, reduction="mean")

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    n_b = len(msgs) // BATCH
    total = n_b * EPOCHS
    step = 0; t0 = time.time()
    for epoch in range(EPOCHS):
        order = np.random.RandomState(SEED + epoch).permutation(len(msgs))
        for b in range(n_b):
            idx = order[b*BATCH:(b+1)*BATCH]
            batch = [wdatum(i) for i in idx]
            lr = LR * max(0.0, 1.0 - step / total)
            fb = tc.forward_backward(batch, loss_fn="cross_entropy")
            op = tc.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
            fbr = fb.result(); op.result()
            if step % 5 == 0 or step == total - 1:
                nll = compute_mean_nll([x["logprobs"] for x in fbr.loss_fn_outputs],
                                       [d.loss_fn_inputs["weights"] for d in batch])
                print(f"[ft] step {step+1}/{total} ep{epoch} lr {lr:.2e} nll {nll:.4f} ({time.time()-t0:.0f}s)")
            step += 1
    path = tc.save_weights_for_sampler(name="rashomon_ft_epr").result().path
    open(PATH_FILE, "w", encoding="utf-8").write(path)
    print(f"[ft] DONE {time.time()-t0:.0f}s -> {path}")
    return path

# ----------------------------- eval ---------------------------------
def evaluate(path, test_df):
    client = oai()
    rows = test_df.reset_index(drop=True)
    preds = [None] * len(rows); samples = []
    def one(i):
        msg = [{"role": "user", "content": pred_user(rows.iloc[i])}]
        for a in range(4):
            try:
                r = client.chat.completions.create(model=path, messages=msg,
                        max_tokens=80, temperature=0.0)
                txt = r.choices[0].message.content or ""
                m = re.findall(r"Answer:\s*([01])", txt)   # answer-first: take the FIRST
                if not m:
                    m = re.findall(r"([01])", txt)
                return i, (int(m[0]) if m else 0), txt
            except Exception:
                time.sleep(2 ** a)
        return i, 0, ""
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for fut in as_completed([ex.submit(one, i) for i in range(len(rows))]):
            i, p, txt = fut.result(); preds[i] = p
            if len(samples) < 3: samples.append(txt)
            done += 1
            if done % 100 == 0:
                print(f"[eval] {done}/{len(rows)} ({time.time()-t0:.0f}s)")
    y = test_df[P.TARGET].values
    preds = np.array([0 if p is None else p for p in preds])
    return P.score(y, preds), preds, samples

# ----------------------------- main ---------------------------------
def main():
    load_tinker_key()
    train_df, test_df = load_split()
    print(f"[data] train={len(train_df)} test={len(test_df)} click_rate={train_df[P.TARGET].mean():.3f}")
    base = baselines(train_df, test_df)
    for k, v in base.items(): print(f"   {k:<8}: {v}")

    rationales = None
    force = os.environ.get("FT_EPR_FORCE_TEACHER", "") not in ("", "0")
    if not force and os.path.exists(RATIONALE_FILE):
        cached = [json.loads(l)["rationale"] for l in open(RATIONALE_FILE, encoding="utf-8")]
        if len(cached) >= len(train_df):
            rationales = cached[:len(train_df)]
            print(f"[teacher] reusing {len(rationales)} cached rationales (FT_EPR_FORCE_TEACHER=1 to regen)")
    if rationales is None and STAGE in ("all", "teacher"):
        rationales = gen_rationales(train_df)
    if STAGE == "teacher":
        print("stage=teacher done"); return
    if rationales is None:
        rationales = [json.loads(l)["rationale"] for l in open(RATIONALE_FILE, encoding="utf-8")][:len(train_df)]

    if STAGE in ("all", "finetune"):
        path = finetune(train_df, rationales)
    else:
        path = open(PATH_FILE, encoding="utf-8").read().strip()
    if STAGE == "finetune":
        print(f"stage=finetune done -> {path}"); return

    print("[eval] self-explaining prediction on test set...")
    ft_score, preds, samples = evaluate(path, test_df)
    print("\n--- sample self-explanations ---")
    for s in samples: print("  •", s.replace("\n", " ")[:200])

    print("\n============= FT + EPR (self-explaining) RESULT =============")
    print(f"  Baseline LogReg     : {base['LogReg']}")
    print(f"  Baseline GBM        : {base['GBM']}")
    print(f"  RashomonLLM FT+EPR  : {ft_score}")
    print(f"  pred balance        : {dict(zip(*np.unique(preds, return_counts=True)))}")
    rec = {"model": BASE_MODEL, "config": {"train": TRAIN_N, "test": TEST_N, "epochs": EPOCHS,
            "batch": BATCH, "lr": LR, "rank": RANK}, "sampler_path": path,
           "baselines": base, "ft_epr": ft_score, "samples": samples}
    json.dump(rec, open(os.path.join(OUT, "ft_epr_result.json"), "w"), indent=2)
    print(f"\nsaved: {os.path.join(OUT, 'ft_epr_result.json')}")

if __name__ == "__main__":
    main()
