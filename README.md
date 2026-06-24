# RashomonLLM — Reproducibility Package (KuaiLive industrial application)

Code to reproduce the **KuaiLive** experiments in *"All Explanations are Wrong, But Many Are
Useful: Exploring the Rashomon Explanations with Large Language Models"* (RashomonLLM): the
**Explanation–Prediction–Reflection (EPR)** agentic workflow and its LoRA-fine-tuned
"Data-Enhanced LLM," evaluated on the KuaiLive live-streaming CTR-prediction task.

> No API keys, no full datasets, and no model checkpoints are included. Supply **your own**
> API token (below), and download the full dataset from the original KuaiLive paper. A small
> 300-row sample is provided under `kuailive/sample_data/` for illustration only.

## Repository layout

```
reproducibility/
├── README.md / RESULTS_MAP.md / requirements.txt / .gitignore
├── data/README.md                 # how to obtain the full KuaiLive dataset
└── kuailive/
    ├── build_ctr_dataset.py / build_ctr_v2.py / build_gift_dataset.py   # dataset builders
    ├── rashomonllm_pilot.py        # in-context EPR loop (Explanation/Prediction/Reflection)
    ├── rashomonllm_finetune.py     # LoRA fine-tune (predict-only)
    ├── rashomonllm_ft_epr.py       # LoRA fine-tune + self-explanation (CoT)
    ├── rashomonllm_std.py          # bootstrap SEs for the prediction & explanation tables
    ├── rashomonllm_expl_quality.py # single-deletion / randomization-check faithfulness
    ├── kuailive_validity.py        # ablations + robustness checks
    ├── kuailive_examples.py        # per-instance natural-language explanation examples
    ├── kuailive_epr_progression.py # EPR explanation-progression trajectory
    ├── rashomonllm_ensemble.py / rashomonllm_epr_correct.py   # supplementary analyses
    ├── sample_data/                # 300-row illustrative excerpt (see its README)
    └── *.local.example             # API-key templates (use your own token)
```

## 1. Environment

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

LoRA fine-tuning (`rashomonllm_finetune.py`, `rashomonllm_ft_epr.py`) uses **Tinker** +
`tinker-cookbook` (Thinking Machines Lab), installed per their instructions. The scikit-learn
baselines and the OpenAI-compatible inference scripts run without them.

## 2. API token (use your own)

Keys are read from an environment variable (preferred) or a git-ignored local file — **supply
your own token; none is bundled.**

```bash
# KuaiLive backbone (Qwen3) is fine-tuned/served via Tinker's OpenAI-compatible endpoint
export TINKER_API_KEY=...        # or: cp kuailive/tinker_key.local.example kuailive/tinker_key.local  (then paste your token)

# Optional: the in-context pilot with PILOT_PROVIDER=openai uses GPT-4o
export OPENAI_API_KEY=sk-...
```

## 3. Data

Download the full **KuaiLive** dataset from the original paper (see
[`data/README.md`](data/README.md)) into `kuailive/raw/`, then build the modeling table.
A 300-row sample for smoke-testing is in `kuailive/sample_data/`.

## 4. Quickstart

Run from `kuailive/`; outputs go to `kuailive/analysis/`.

```bash
cd kuailive
python build_ctr_v2.py                      # -> analysis/ctr_dataset_v2.csv (needs raw/ data)

export TINKER_API_KEY=...
export FT_DATA="analysis/ctr_dataset_v2.csv"
python rashomonllm_finetune.py              # predict-only LoRA  (-> analysis/ft_sampler_path.txt)
python rashomonllm_ft_epr.py                # + self-explanation (CoT)

python rashomonllm_std.py                   # prediction & explanation tables (mean ± bootstrap SE)
python rashomonllm_expl_quality.py          # faithfulness (single deletion / randomization)
python kuailive_validity.py                 # feature ablation, CTR baselines, robustness checks
python kuailive_examples.py                 # per-instance explanation examples
python kuailive_epr_progression.py          # EPR progression (in-context)
```

A full mapping from each paper table to its script and output is in
[`RESULTS_MAP.md`](RESULTS_MAP.md).

## 5. Configuration knobs

Settings are environment variables read at the top of each script: `FT_DATA`, `FT_MODEL`,
`FT_TRAIN`/`FT_EPOCHS`/`FT_RANK` (and `FT_EPR_*`), `PILOT_PROVIDER`/`PILOT_MODEL`,
`KUAILIVE_RAW`. The backbone / hyperparameter / data-scale ablations in the paper are
produced by re-running the fine-tuning scripts with the corresponding overrides.

## 6. Notes

- **LLM non-determinism.** Greedy decoding (`temperature=0`) is used for predictions, but
  exact numbers can vary slightly across model/endpoint versions; the robustness checks
  (temperature, seeds, feature order) quantify this.
- **Backbone.** KuaiLive uses `Qwen/Qwen3-30B-A3B-Instruct-2507` via Tinker.
- **Leakage safety.** The collaborative features in `build_ctr_v2.py` are computed from a
  history window that strictly precedes the labeled exposures (see comments in that file).
