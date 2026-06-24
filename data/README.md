# Dataset — KuaiLive

The full dataset is **not** redistributed here. Download it from the original source and place
it as indicated. A 300-row illustrative sample is included separately under
`../kuailive/sample_data/` (see that folder's README).

## Source

- **KuaiLive** — https://imgkkk574.github.io/KuaiLive
  (Qu et al., 2026, *KuaiLive: A Real-time Interactive Dataset for Live Streaming
  Recommendation*, SIGIR 2026).

## Setup

1. Download the raw KuaiLive CSVs (`click.csv`, `negative.csv`, `user.csv`, `streamer.csv`,
   `room.csv`, `gift.csv`, …) and place them in **`kuailive/raw/`** (or point the
   `KUAILIVE_RAW` environment variable at the folder that contains them).
2. Build the modeling tables (written to `kuailive/analysis/`):
   - `build_ctr_dataset.py` → `analysis/ctr_dataset.csv`     (28 static, interpretable features)
   - `build_ctr_v2.py`      → `analysis/ctr_dataset_v2.csv`  (adds leakage-safe, time-aware
     collaborative features: historical user / streamer / category CTR + activity counts)

To explore the feature format or smoke-test the pipeline without the full download, use the
300-row excerpt in `kuailive/sample_data/`.
