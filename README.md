# toronto-bikeshare-ml

Pipeline for transforming Bike Share Toronto trip exports into station-level demand series and training a GRU + station embedding model to forecast next-day trip counts.

## Overview
- End-to-end workflow: ingest raw monthly CSVs, clean and localize timestamps, aggregate daily station demand, and fit a PyTorch GRU forecaster.
- Purpose-built for operational planning: surfaces station-level demand projections and error diagnostics so planners can spot high-variance docks.
- Robust defaults: timezone handling, encoding detection, data quality filters, and rich evaluation reports ship with the standard pipeline.

## Current Results Snapshot (2025-10-03)
Latest full run on the complete public export baseline:
- Data volume after cleaning: 19,249,900 trips covering 878 stations with ≥60 days of history.
- Global validation metrics (30-day holdout, 1-day horizon): loss=0.1547 (log space), rmse=12.10 trips, mae=6.91 trips.
- Error quantiles (absolute trips): p50=3.53, p75=8.98, p90=17.45, p95=24.80, max=268.79.
- Lowest MAE stations (top 5): 7481, 7697, 7177, 7509, 7472 — each ≈0.01 MAE over 90 samples.
- Highest MAE stations (top 5): 7016 (36.6), 7006 (38.7), 7019 (39.0), 7076 (45.5), 7171 (46.9) trips MAE.
- Random validation samples illustrate typical residuals, e.g. station 7430 predicted 75.4 vs actual 85.0 (|Δ|=9.6).

Re-run the evaluator whenever you refresh data, adjust hyperparameters, or retrain—metrics are cached only in the emitted logs and `artifacts/metrics.json`.

## Key Components
- `prep_and_train.py`: Orchestrates data preparation, feature engineering, and GRU training.
- `evaluate_model.py`: Reloads the trained artifact, rebuilds the validation window, and issues aggregate plus per-station diagnostics.
- `artifacts/`: Default location for serialized model state (`station_demand_gru.pt`) and validation metrics (`metrics.json`).

## Data Requirements
- Python 3.9+ environment with `pip`.
- Local Bike Share Toronto ridership exports placed under `data/bikeshare-ridership-*/` (each directory contains the official monthly CSVs).

## Environment Setup
Install dependencies in your active virtualenv or Conda env:
```
pip install -r requirements.txt
```

## Training Pipeline
Run the full preparation and training pipeline:
```
python3 prep_and_train.py --data-root data --output-dir artifacts
```

Useful flags:
- `--history-days` (default 14): trailing daily observations fed to the GRU.
- `--forecast-horizon` (default 1): number of days ahead to predict.
- `--eval-days` (default 30): length of the holdout validation window.
- `--min-station-days` (default 60): minimum history to keep a station.

If your environment blocks OpenMP shared memory, prefix with:
```
KMP_CREATE_SHM=0 KMP_INIT_AT_FORK=FALSE OMP_NUM_THREADS=1 python3 prep_and_train.py
```
to suppress Intel OpenMP runtime errors.

## Evaluation Procedure
1. Ensure training artifacts exist (see `artifacts/` after running `prep_and_train.py`).
2. Recreate the validation split and diagnostics:
   ```
   python3 evaluate_model.py --top-k 10 --num-samples 10
   ```
3. Review console output for:
   - Global loss / RMSE / MAE on the holdout window.
   - Quantiles of absolute error across all station-day predictions.
   - Top/bottom stations by MAE and RMSE to pinpoint consistently mis-modeled docks.
   - Random sample predictions to spot systematic bias or outliers.
4. Optional: pass `--output metrics_latest.json` to persist a fresh metrics file alongside the logs.

## Operational Notes
- Timestamps are localized to `America/Toronto` so midnight boundaries and DST changes align with local service days.
- CSV ingestion auto-detects encodings (via `charset-normalizer`) and falls back gracefully instead of dropping months with smart quotes or Windows-1252 characters.
- Daily counts are zero-filled for missing station/date pairs before sequence construction to avoid leakage.
- The GRU learns a shared temporal pattern complemented by station embeddings and time-of-week covariates for better per-dock calibration.
