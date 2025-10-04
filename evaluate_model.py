#!/usr/bin/env python3
"""Evaluate a trained Bike Share Toronto demand model on the validation split."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Avoid OpenMP shared-memory usage in sandboxed environments.
os.environ.setdefault("KMP_CREATE_SHM", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prep_and_train import (
    aggregate_daily_counts,
    build_feature_arrays,
    choose_device,
    evaluate,
    expand_station_timeseries,
    filter_min_history,
    find_csv_files,
    load_all_trips,
    make_datasets,
    StationDemandGRU,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/station_demand_gru.pt"),
        help="Path to the saved model artifact (default: artifacts/station_demand_gru.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing bikeshare CSV exports (default: data)",
    )
    parser.add_argument(
        "--min-station-days",
        type=int,
        default=60,
        help="Minimum days of history required per station (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for evaluation DataLoader (default: 4096)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample predictions to display from the validation split (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of stations to list for best/worst MAE (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (cpu, cuda, mps). Defaults to auto-detect.",
    )
    return parser.parse_args()


def load_artifact(path: Path) -> Tuple[Dict, Dict[str, int], Dict[str, torch.Tensor]]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Artifact missing required keys: expected 'config' and 'model_state_dict'")
    return checkpoint["config"], checkpoint["station_to_idx"], checkpoint


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    config, station_to_idx_saved, checkpoint = load_artifact(args.artifact)

    history_days = config["history_days"]
    forecast_horizon = config["forecast_horizon"]
    eval_days = config["eval_days"]
    feature_dim = config["feature_dim"]
    hidden_size = config["hidden_size"]
    embedding_dim = config["embedding_dim"]
    num_stations = config["num_stations"]

    device = (
        torch.device(args.device)
        if args.device is not None
        else choose_device()
    )
    logging.info("Evaluating on device: %s", device)

    csv_paths = find_csv_files(args.data_root)
    trips = load_all_trips(csv_paths)

    daily = aggregate_daily_counts(trips)
    daily = filter_min_history(daily, args.min_station_days)
    if daily.empty:
        raise RuntimeError("No stations remaining after min history filter")

    expanded = expand_station_timeseries(daily)
    station_ids_raw, _, sequences_raw, targets_raw, dates_raw = build_feature_arrays(expanded)

    available_station_ids = {sid for sid in station_ids_raw}
    missing_stations = [sid for sid in station_to_idx_saved if sid not in available_station_ids]
    if missing_stations:
        logging.warning(
            "Skipping %d stations absent from current data snapshot", len(missing_stations)
        )

    station_ids = [sid for sid in station_to_idx_saved if sid in available_station_ids]
    if not station_ids:
        raise RuntimeError("No overlapping stations between artifact and current data")

    station_to_idx = {sid: station_to_idx_saved[sid] for sid in station_ids}

    sequences = {sid: sequences_raw[sid] for sid in station_ids}
    targets = {sid: targets_raw[sid] for sid in station_ids}
    dates = {sid: dates_raw[sid] for sid in station_ids}

    train_ds, val_ds = make_datasets(
        station_ids,
        station_to_idx,
        sequences,
        targets,
        dates,
        history_days=history_days,
        forecast_horizon=forecast_horizon,
        eval_days=eval_days,
    )

    if val_ds is None or len(val_ds) == 0:
        raise RuntimeError("Validation split is empty; unable to evaluate model")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = StationDemandGRU(
        num_stations=num_stations,
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, val_loader, torch.nn.MSELoss(), device)
    logging.info(
        "Validation metrics - loss=%.4f rmse=%.2f mae=%.2f",
        metrics["loss"],
        metrics["rmse"],
        metrics["mae"],
    )

    idx_to_station = {idx: sid for sid, idx in station_to_idx_saved.items()}
    rows: List[Dict[str, float]] = []
    with torch.no_grad():
        for sequences_batch, station_idx_batch, target_batch in val_loader:
            sequences_batch = sequences_batch.to(device)
            station_idx_batch = station_idx_batch.to(device)
            target_batch = target_batch.to(device)

            preds_batch = model(sequences_batch, station_idx_batch)

            preds_counts = np.expm1(preds_batch.cpu().numpy())
            target_counts = np.expm1(target_batch.cpu().numpy())
            station_indices = station_idx_batch.cpu().numpy()

            for sid_idx, pred, actual in zip(station_indices, preds_counts, target_counts):
                station_id = idx_to_station.get(int(sid_idx), f"idx_{sid_idx}")
                rows.append(
                    {
                        "station_id": station_id,
                        "station_idx": int(sid_idx),
                        "prediction": float(pred),
                        "actual": float(actual),
                        "error": float(pred - actual),
                        "abs_error": float(abs(pred - actual)),
                    }
                )
    if not rows:
        logging.warning("No validation predictions collected")
        return

    frame = pd.DataFrame(rows)
    abs_err = frame["abs_error"].to_numpy()
    logging.info(
        "Error quantiles (counts) - p50=%.2f p75=%.2f p90=%.2f p95=%.2f max=%.2f",
        np.quantile(abs_err, 0.5),
        np.quantile(abs_err, 0.75),
        np.quantile(abs_err, 0.9),
        np.quantile(abs_err, 0.95),
        abs_err.max(),
    )

    per_station = (
        frame.groupby("station_id")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            actual_mean=("actual", "mean"),
            actual_median=("actual", "median"),
            samples=("actual", "size"),
        )
        .sort_values("mae")
    )

    top_k = max(args.top_k, 1)
    logging.info("Lowest MAE stations (top %d):", top_k)
    for station_id, row in per_station.head(top_k).iterrows():
        logging.info(
            "station=%s mae=%.2f rmse=%.2f avg_actual=%.2f samples=%d",
            station_id,
            row.mae,
            row.rmse,
            row.actual_mean,
            int(row.samples),
        )

    logging.info("Highest MAE stations (top %d):", top_k)
    for station_id, row in per_station.tail(top_k).iterrows():
        logging.info(
            "station=%s mae=%.2f rmse=%.2f avg_actual=%.2f samples=%d",
            station_id,
            row.mae,
            row.rmse,
            row.actual_mean,
            int(row.samples),
        )

    samples_to_show = max(args.num_samples, 0)
    if samples_to_show:
        sample_rows = frame.sample(n=min(samples_to_show, len(frame)), random_state=0)
        logging.info("Random validation samples:")
        for _, row in sample_rows.iterrows():
            logging.info(
                "station=%s pred=%.1f actual=%.1f abs_error=%.1f",
                row.station_id,
                row.prediction,
                row.actual,
                row.abs_error,
            )


if __name__ == "__main__":
    main()
