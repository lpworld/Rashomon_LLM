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
| Prediction-performance table (Acc / F1 ± bootstrap SE) | `rashomonllm_std.py` **(needs FT)** | `analysis/std_result.json` |
| Explanation-quality table (Single Deletion, Randomization Check) | `rashomonllm_std.py`, `rashomonllm_expl_quality.py` **(needs FT)** | `analysis/std_result.json`, `analysis/expl_quality_result.json` |
| Per-instance explanation examples | `kuailive_examples.py` **(needs FT)** | `analysis/explanation_examples.json` |
| EPR explanation-progression table | `kuailive_epr_progression.py` | `analysis/epr_progression.json` |
| Feature-group ablation, specialized-CTR-baseline comparison, robustness checks | `kuailive_validity.py` **(needs FT)** | `analysis/validity_result.json` |
| Ensemble analysis (LLM + GBM) | `rashomonllm_ensemble.py` **(needs FT)** | `analysis/ensemble_result.json` |
| EPR-correction analysis (reflection on a saturated predictor) | `rashomonllm_epr_correct.py` **(needs FT)** | `analysis/epr_correct_result.json` |

Notes:
- The **method-component / backbone / hyperparameter / data-scale ablations** are produced by
  re-running `rashomonllm_finetune.py` / `rashomonllm_ft_epr.py` with the corresponding
  environment overrides (see the `os.environ.get(...)` knobs at the top of each script).
- Local (non-LLM) baselines (Logistic Regression, Gradient-Boosted Trees) are computed inside
  `rashomonllm_std.py` / `kuailive_validity.py` with scikit-learn.
