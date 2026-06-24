# -*- coding: utf-8 -*-
"""
RashomonLLM "Data-Enhanced LLM" via real LoRA fine-tuning on Tinker (KuaiLive CTR).

Instead of cramming labeled rows into a 32K context (the in-context pilot's bottleneck),
this injects the full training set into the model weights with LoRA SFT, then evaluates
the fine-tuned model on a held-out test set and compares to local ML baselines.

Phases:
  TRAIN : create LoRA client (Qwen3-30B-A3B-Instruct-2507), SFT on (features -> 0/1),
          save sampler weights -> tinker:// path (written to ft_sampler_path.txt).
  EVAL  : per-row prediction via the OpenAI-compatible endpoint pointed at that path
          (concurrent), compute Accuracy / macro-F1, compare to LogReg + GBM baselines.

Runs on your Tinker (TML) credit. Key read from tinker_key.local (or TINKER_API_KEY env).
Re-run EVAL only:  set FT_EVAL_ONLY=1  (reads ft_sampler_path.txt).
"""
import os, sys, re, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rashomonllm_pilot as P   # reuse serialize(), FEATURES, TARGET, score()

DATA = os.environ.get("FT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "ctr_dataset.csv"))
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis")
PATH_FILE = os.path.join(OUT, "ft_sampler_path.txt")
TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

def _envi(n, d): return int(os.environ.get(n, d))

FT_MODEL   = os.environ.get("FT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
FT_TRAIN_N = _envi("FT_TRAIN", 8000)
FT_TEST_N  = _envi("FT_TEST", 1000)
FT_EPOCHS  = _envi("FT_EPOCHS", 2)
FT_BATCH   = _envi("FT_BATCH", 128)
FT_LR      = float(os.environ.get("FT_LR", "1e-4"))
FT_RANK    = _envi("FT_RANK", 32)
FT_CONC    = _envi("FT_CONC", 8)
FT_MAXLEN  = 2048
EVAL_ONLY  = os.environ.get("FT_EVAL_ONLY", "") not in ("", "0", "false", "False")
SEED       = 42

PROMPT_PREFIX = ("You are predicting user engagement on a live-streaming platform.\n"
                 "Exposure features:\n")
QUESTION = ("\n\nWill this user click into this live room? "
            "Answer with a single digit: 1 = click, 0 = no click.")

def to_messages(row):
    return [{"role": "user", "content": PROMPT_PREFIX + P.serialize(row) + QUESTION},
            {"role": "assistant", "content": str(int(row[P.TARGET]))}]

def to_user_prompt(row):
    return PROMPT_PREFIX + P.serialize(row) + QUESTION

# ----------------------------- key ----------------------------------
def load_tinker_key():
    if os.environ.get("TINKER_API_KEY"):
        return
    kf = os.path.join(HERE, "tinker_key.local")
    if os.path.exists(kf):
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

# ----------------------------- data ----------------------------------
def load_split():
    df = pd.read_csv(DATA).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # derive feature list dynamically (supports v1 static and v2 +historical) and
    # patch it into the shared serializer so to_messages/eval use the right columns
    P.FEATURES = [c for c in df.columns if c not in ("user_id", "live_id", "streamer_id", P.TARGET)]
    print(f"[features] {len(P.FEATURES)} features: {', '.join(P.FEATURES)}")
    train = df.iloc[:FT_TRAIN_N].reset_index(drop=True)
    test  = df.iloc[FT_TRAIN_N:FT_TRAIN_N + FT_TEST_N].reset_index(drop=True)
    return train, test

def baselines(train, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    comb = pd.concat([train, test], ignore_index=True)
    # keep continuous/high-cardinality columns numeric; one-hot the categorical buckets
    num = [c for c in P.FEATURES if pd.api.types.is_numeric_dtype(comb[c]) and comb[c].nunique() > 30]
    cat = [c for c in P.FEATURES if c not in num]
    X = pd.concat([pd.get_dummies(comb[cat].astype(str)),
                   comb[num].reset_index(drop=True)], axis=1)
    Xtr, Xte = X.iloc[:len(train)], X.iloc[len(train):]
    ytr, yte = train[P.TARGET].values, test[P.TARGET].values
    res = {}
    res["LogReg"] = P.score(yte, LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict(Xte))
    res["GBM"]    = P.score(yte, HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr).predict(Xte))
    return res

# ----------------------------- train ---------------------------------
def train_lora(train_df):
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.supervised.data import conversation_to_datum
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tok = get_tokenizer(FT_MODEL)
    rname = model_info.get_recommended_renderer_name(FT_MODEL)
    renderer = renderers.get_renderer(rname, tok)
    print(f"[train] model={FT_MODEL} renderer={rname} rank={FT_RANK} "
          f"rows={len(train_df)} batch={FT_BATCH} epochs={FT_EPOCHS}")

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=FT_MODEL, rank=FT_RANK)

    n_batches = len(train_df) // FT_BATCH
    total_steps = n_batches * FT_EPOCHS
    step = 0
    t0 = time.time()
    for epoch in range(FT_EPOCHS):
        order = np.random.RandomState(SEED + epoch).permutation(len(train_df))
        for b in range(n_batches):
            idx = order[b * FT_BATCH:(b + 1) * FT_BATCH]
            batch = [conversation_to_datum(to_messages(train_df.iloc[i]), renderer,
                                           FT_MAXLEN, renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES)
                     for i in idx]
            lr = FT_LR * max(0.0, 1.0 - step / total_steps)
            try:
                fb = tc.forward_backward(batch, loss_fn="cross_entropy")
                op = tc.optim_step(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
                fbr = fb.result(); op.result()
            except Exception as e:
                print(f"[train] step {step+1} transient FAIL ({type(e).__name__}); skipping")
                step += 1
                continue
            if step % 5 == 0 or step == total_steps - 1:
                logprobs = [x["logprobs"] for x in fbr.loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in batch]
                nll = compute_mean_nll(logprobs, weights)
                print(f"[train] step {step+1}/{total_steps} epoch {epoch} lr {lr:.2e} "
                      f"nll {nll:.4f}  ({time.time()-t0:.0f}s)")
            step += 1

    path = tc.save_weights_for_sampler(name="rashomon_ctr_ft").result().path
    with open(PATH_FILE, "w", encoding="utf-8") as f:
        f.write(path)
    print(f"[train] DONE in {time.time()-t0:.0f}s  sampler path -> {path}")
    return path

# ----------------------------- eval ----------------------------------
def evaluate(path, test_df):
    from openai import OpenAI
    client = OpenAI(base_url=TINKER_BASE_URL, api_key=os.environ["TINKER_API_KEY"])
    rows = test_df.reset_index(drop=True)
    preds = [None] * len(rows)

    def one(i):
        msg = [{"role": "user", "content": to_user_prompt(rows.iloc[i])}]
        for attempt in range(4):
            try:
                r = client.chat.completions.create(model=path, messages=msg,
                                                   max_tokens=8, temperature=0.0)
                txt = r.choices[0].message.content or ""
                m = re.search(r"[01]", txt)
                return i, (int(m.group()) if m else 0)
            except Exception:
                time.sleep(2 ** attempt)
        return i, 0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=FT_CONC) as ex:
        futs = [ex.submit(one, i) for i in range(len(rows))]
        for fut in as_completed(futs):
            i, p = fut.result(); preds[i] = p
            done += 1
            if done % 100 == 0:
                print(f"[eval] {done}/{len(rows)}  ({time.time()-t0:.0f}s)")
    y = test_df[P.TARGET].values
    preds = np.array([0 if p is None else p for p in preds])
    return P.score(y, preds), preds

# ----------------------------- main ----------------------------------
def main():
    load_tinker_key()
    train_df, test_df = load_split()
    print(f"[data] train={len(train_df)} test={len(test_df)}  "
          f"train click rate={train_df[P.TARGET].mean():.3f}")

    print("[baselines] computing (free)...")
    base = baselines(train_df, test_df)
    for k, v in base.items():
        print(f"   {k:<8}: {v}")

    if EVAL_ONLY:
        path = open(PATH_FILE, encoding="utf-8").read().strip()
        print(f"[eval-only] using {path}")
    else:
        path = train_lora(train_df)

    print("[eval] predicting on test set via fine-tuned model...")
    ft_score, preds = evaluate(path, test_df)

    print("\n================= FINE-TUNE RESULT =================")
    print(f"  Baseline LogReg : {base['LogReg']}")
    print(f"  Baseline GBM    : {base['GBM']}")
    print(f"  RashomonLLM-FT  : {ft_score}")
    print(f"  pred balance    : {dict(zip(*np.unique(preds, return_counts=True)))}")

    rec = {"model": FT_MODEL,
           "config": {"train": FT_TRAIN_N, "test": FT_TEST_N, "epochs": FT_EPOCHS,
                      "batch": FT_BATCH, "lr": FT_LR, "rank": FT_RANK},
           "sampler_path": path, "baselines": base, "finetune": ft_score}
    with open(os.path.join(OUT, "finetune_result.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"\nsaved: {os.path.join(OUT, 'finetune_result.json')}")

if __name__ == "__main__":
    main()
