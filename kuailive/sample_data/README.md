# Illustrative data sample

`ctr_dataset_v2.sample.csv` is a **small 300-row excerpt** of the modeling table produced by
`build_ctr_v2.py`. It is included only to illustrate the feature schema and to let you
smoke-test the pipeline end-to-end without downloading the full dataset.

Columns: `user_id, live_id, streamer_id, clicked` + 28 static attributes + 5 time-aware
collaborative features (`user_ctr, user_n, streamer_ctr, streamer_n, cat_ctr`).

**It is NOT sufficient to reproduce the paper's numbers** (those use the full ~60k balanced
sample drawn from the complete KuaiLive log). To reproduce the reported results:

1. Download the full **KuaiLive** dataset from the original paper —
   https://imgkkk574.github.io/KuaiLive (Qu et al., 2026) — into `../raw/`.
2. Run `python ../build_ctr_v2.py` to regenerate the full `../analysis/ctr_dataset_v2.csv`.

To smoke-test the downstream scripts on this sample instead:

```bash
mkdir -p ../analysis && cp ctr_dataset_v2.sample.csv ../analysis/ctr_dataset_v2.csv
```
