# Mapping: Paper Results → Scripts → Output Files (KuaiLive, Section 6)

Tables are referenced by content (numbers shift between revisions). Items marked **(needs FT)**
require a fine-tuned checkpoint first — run `rashomonllm_finetune.py` (and/or
`rashomonllm_ft_epr.py`), which writes the sampler path to `analysis/ft_sampler_path.txt`,
before the downstream evaluation scripts. All outputs are written under `kuailive/analysis/`.

| Paper result | Script | Output |
|---|---|---|
| Build modeling tables (static / +collaborative) | `build_ctr_dataset.py`, `build_ctr_v2.py` | `analysis/ctr_dataset.csv`, `analysis/ctr_dataset_v2.csv` |
| In-context EPR pilot (one-shot → +reflection) | `rashomonllm_pilot.py` | `analysis/pilot_result.json` |
| Fine-tune RashomonLLM (predict-only LoRA) | `rashomonllm_finetune.py` | `analysis/finetune_result.json`, `analysis/ft_sampler_path.txt` |
| Fine-tune + self-explanation (EPR CoT) | `rashomonllm_ft_epr.py` | `analysis/ft_epr_result.json`, `analysis/ft_epr_sampler_path.txt` |
| Placebo-rationale control (matched-length, fidelity-destroyed) | `rashomonllm_ft_epr.py` with `FT_EPR_PLACEBO=1` | `analysis/ft_epr_result.json` |
| Prediction-performance table (Acc / F1 ± bootstrap SE) | `rashomonllm_std.py` **(needs FT)** | `analysis/std_result.json` |
| Explanation-quality table (Single Deletion, Randomization Check) | `rashomonllm_std.py`, `rashomonllm_expl_quality.py` **(needs FT)** | `analysis/std_result.json`, `analysis/expl_quality_result.json` |
| Per-instance explanation examples | `kuailive_examples.py` **(needs FT)** | `analysis/explanation_examples.json` |
| EPR explanation-progression table | `kuailive_epr_progression.py` | `analysis/epr_progression.json` |
| Feature-group ablation, specialized-CTR-baseline comparison, robustness checks | `kuailive_validity.py` **(needs FT)** | `analysis/validity_result.json` |
| Ensemble analysis (LLM + GBM) | `rashomonllm_ensemble.py` **(needs FT)** | `analysis/ensemble_result.json` |
| EPR-correction analysis (reflection on a saturated predictor) | `rashomonllm_epr_correct.py` **(needs FT)** | `analysis/epr_correct_result.json` |
| Subgroup fairness audit + equal-opportunity recalibration (Table `tbl:kuaifair`) | `audit_fair.py` (acc/FPR/FNR by group), `audit_fair_recalib.py` (adds per-group threshold recalibration) **(needs FT)** | console (`AUDIT_RESULT` lines / baseline-vs-recalibrated spreads) |

Notes:
- The **method-component / backbone / hyperparameter / data-scale ablations** are produced by
  re-running `rashomonllm_finetune.py` / `rashomonllm_ft_epr.py` with the corresponding
  environment overrides (see the `os.environ.get(...)` knobs at the top of each script).
- The **placebo-rationale control** (paper Table `tbl:kuaicomp`) reuses `rashomonllm_ft_epr.py`
  with `FT_EPR_PLACEBO=1`: it keeps the genuine teacher rationales (fixing output length and
  surface form) but shuffles them across instances so fidelity is destroyed. Accuracy then
  collapsing to the predict-only level isolates the gain to rationale *fidelity*, not token count.
- The **Rashomon-set-size sweep on KuaiLive** (paper Table `tbl:kuaiksweep`) and the
  **deployment-cost table** (`tbl:kuaicost`) are not yet separate scripts: the set-size sweep
  re-runs the in-context aggregation over `k ∈ {1,10,25,50,100}` retained explanations, and the
  cost figures (LoRA GPU-hours, ms/prediction, $/1k) are read off the fine-tuning and evaluation
  wall-clock on your serving hardware.
- Local (non-LLM) baselines (Logistic Regression, Gradient-Boosted Trees) are computed inside
  `rashomonllm_std.py` / `kuailive_validity.py` with scikit-learn.
- The **subgroup fairness audit** (paper Table `tbl:kuaifair`) scores a held-out window (rows
  30000:32500 of `ctr_dataset_v2.csv`, disjoint from training) with the deployed predict-only
  sampler in `analysis/ft_sampler_path.txt`. `audit_fair.py` reports accuracy / FPR / FNR by
  `u_gender` and `u_age` at the global 0.5 threshold; `audit_fair_recalib.py` additionally scores a
  calibrated `P(click)` via the chosen-token logprob and applies **equal-opportunity** post-processing
  (Hardt, Price & Srebro 2016) — a per-group decision threshold that equalizes the true-positive rate —
  then re-reports per-group FPR/FNR and the max-minus-min spreads. Recalibration drives the FNR spread
  to ~0 and roughly halves the residual FPR gap at a small accuracy cost; numbers are read from the
  console output. No API keys are stored in either script (credentials load via `load_tinker_key()`).
