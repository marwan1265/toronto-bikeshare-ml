#!/usr/bin/env python3
"""Prep data and train a GRU with station embeddings for Bike Share Toronto."""

from __future__ import annotations

import os

# Force OpenMP to avoid shared-memory segments that are unavailable in
# constrained/sandboxed environments.
os.environ.setdefault("KMP_CREATE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


START_TIME_ALIASES = [
    "start_time",
    "trip_start_time",
    "start_time_local",
    "start_time_utc",
    "trip_start_timestamp",
    "start_trip_time",
]
START_STATION_ALIASES = [
    "start_station_id",
    "from_station_id",
    "start_station_code",
    "start_station",
]
USER_TYPE_ALIASES = [
    "user_type",
    "usertype",
    "member_type",
    "passholder_type",
    "usercategory",
]

TORONTO_TZ = "America/Toronto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing bikeshare CSV exports (default: data)",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=14,
        help="Number of trailing days to feed into the GRU (default: 14)",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="How many days ahead to predict (default: 1 for next-day)",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=90,
        help="Number of most-recent target days to reserve for validation (default: 30)",
    )
    parser.add_argument(
        "--min-station-days",
        type=int,
        default=60,
        help="Require at least this many days of history per station before using it (default: 60)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for model training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs (default: 10)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Hidden size for GRU (default: 64)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding size per station (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where artifacts (model, metrics) will be stored (default: artifacts)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "manual_seed"):
            torch.mps.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data root '{root}' does not exist")
    paths = sorted(root.glob("bikeshare-ridership-*/*.csv"))
    logging.info("Found %d CSV files under %s", len(paths), root)
    return paths


def _standardize_column_names(columns: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in columns:
        normalized = col.strip().lower().replace("/", "_").replace(" ", "_")
        normalized = normalized.replace("__", "_")
        mapping[col] = normalized
    return mapping


def _pick_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def load_trip_frame(csv_path: Path) -> pd.DataFrame:
    logging.debug("Loading %s", csv_path)
    df = _read_csv_autodetect(csv_path)
    rename_map = _standardize_column_names(df.columns)
    df = df.rename(columns=rename_map)

    start_time_col = _pick_column(df.columns, START_TIME_ALIASES)
    station_id_col = _pick_column(df.columns, START_STATION_ALIASES)
    if start_time_col is None or station_id_col is None:
        missing = []
        if start_time_col is None:
            missing.append("start time column")
        if station_id_col is None:
            missing.append("start station id column")
        raise ValueError(f"Required columns missing in {csv_path}: {', '.join(missing)}")

    selected_cols = [start_time_col, station_id_col]
    user_type_col = _pick_column(df.columns, USER_TYPE_ALIASES)
    if user_type_col:
        selected_cols.append(user_type_col)

    rename_args = {
        start_time_col: "start_time",
        station_id_col: "start_station_id",
    }
    if user_type_col:
        rename_args[user_type_col] = "user_type"

    df = df[selected_cols].rename(columns=rename_args)

    df["start_time"] = _to_toronto_time(df["start_time"], start_time_col)
    df = df.dropna(subset=["start_time", "start_station_id"])
    df["start_station_id"] = df["start_station_id"].astype(str).str.strip()
    df = df[df["start_station_id"] != ""]

    if "user_type" in df.columns:
        df["user_type"] = (
            df["user_type"].astype(str).str.strip().str.lower().replace("", np.nan)
        )
        df["user_type"] = df["user_type"].apply(_canonicalize_user_type)

    return df


def _read_csv_autodetect(csv_path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "utf-8"]
    last_error: Optional[Exception] = None
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            logging.debug(
                "Failed to read %s with encoding %s: %s",
                csv_path,
                encoding,
                exc,
            )

    detected = _detect_encoding(csv_path)
    if detected:
        try:
            logging.info("Detected encoding '%s' for %s", detected, csv_path.name)
            return pd.read_csv(csv_path, encoding=detected, encoding_errors="replace")
        except UnicodeDecodeError as exc:
            last_error = exc
            logging.debug(
                "Detected encoding '%s' failed for %s: %s",
                detected,
                csv_path,
                exc,
            )

    logging.warning(
        "Falling back to latin-1 for %s after decode errors (%s)",
        csv_path,
        last_error,
    )
    return pd.read_csv(csv_path, encoding="latin-1", encoding_errors="replace")


def _detect_encoding(csv_path: Path) -> Optional[str]:
    try:
        from charset_normalizer import from_path
    except ImportError:
        logging.debug("charset_normalizer not available; skipping encoding detection")
        return None

    result = from_path(str(csv_path)).best()
    if result is None or result.encoding is None:
        return None
    return result.encoding


def _to_toronto_time(series: pd.Series, source_column: str) -> pd.Series:
    if source_column.endswith("_utc"):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert(TORONTO_TZ)

    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if parsed.dt.tz is not None:
        return parsed.dt.tz_convert(TORONTO_TZ)

    return parsed.dt.tz_localize(TORONTO_TZ, nonexistent="shift_forward", ambiguous="NaT")


def _canonicalize_user_type(raw: Optional[str]) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if "casual" in value:
        return "casual"
    if "member" in value:
        return "member"
    return value


def load_all_trips(csv_paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        try:
            frames.append(load_trip_frame(path))
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.warning("Skipping %s due to %s", path, exc)
    if not frames:
        raise RuntimeError("No valid CSV files could be loaded")
    combined = pd.concat(frames, ignore_index=True)
    logging.info("Loaded %d trips after cleaning", len(combined))
    return combined


def aggregate_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    toronto_times = df["start_time"].dt.tz_convert(TORONTO_TZ)
    df["trip_date"] = toronto_times.dt.normalize().dt.tz_localize(None)
    base = (
        df.groupby(["start_station_id", "trip_date"]).size().rename("trip_count").reset_index()
    )

    if "user_type" in df.columns:
        user_counts = (
            df.groupby(["start_station_id", "trip_date", "user_type"]).size().rename("count").reset_index()
        )
        user_pivot = user_counts.pivot_table(
            index=["start_station_id", "trip_date"],
            columns="user_type",
            values="count",
            fill_value=0,
        )
        user_pivot.columns = [f"user_count_{col}" for col in user_pivot.columns]
        user_pivot = user_pivot.reset_index()
        base = base.merge(user_pivot, on=["start_station_id", "trip_date"], how="left")

    return base


def filter_min_history(daily: pd.DataFrame, min_days: int) -> pd.DataFrame:
    counts = daily.groupby("start_station_id")["trip_date"].nunique()
    keep_ids = counts[counts >= min_days].index
    filtered = daily[daily["start_station_id"].isin(keep_ids)].copy()
    logging.info(
        "Retained %d stations with >= %d days of data",
        filtered["start_station_id"].nunique(),
        min_days,
    )
    return filtered


def expand_station_timeseries(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["start_station_id", "trip_date"]).copy()
    min_date = daily["trip_date"].min()
    max_date = daily["trip_date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")

    frames: List[pd.DataFrame] = []
    feature_cols = [col for col in daily.columns if col not in {"start_station_id", "trip_date"}]
    for station_id, group in daily.groupby("start_station_id"):
        indexed = group.set_index("trip_date").reindex(all_dates, fill_value=np.nan)
        indexed.index.name = "trip_date"
        indexed = indexed.reset_index()
        indexed["start_station_id"] = station_id
        # fill missing counts with zero, other features with zero as a reasonable default
        indexed["trip_count"] = indexed["trip_count"].fillna(0)
        for col in feature_cols:
            if col != "trip_count":
                indexed[col] = indexed[col].fillna(0)
        frames.append(indexed)
    expanded = pd.concat(frames, ignore_index=True)
    return expanded


def build_feature_arrays(
    timeseries: pd.DataFrame,
) -> Tuple[
    List[str],
    Dict[str, int],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, pd.Series],
]:
    timeseries = timeseries.sort_values(["start_station_id", "trip_date"]).copy()
    station_ids = timeseries["start_station_id"].unique().tolist()
    station_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}

    grouped = timeseries.groupby("start_station_id")
    sequences_per_station: Dict[str, np.ndarray] = {}
    targets_per_station: Dict[str, np.ndarray] = {}
    dates_per_station: Dict[str, pd.Series] = {}

    for station_id, group in grouped:
        group = group.reset_index(drop=True)
        counts = group["trip_count"].to_numpy(dtype=np.float32)
        log_counts = np.log1p(counts)
        dow = group["trip_date"].dt.dayofweek.to_numpy(dtype=np.float32)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)

        features = np.stack([log_counts, dow_sin, dow_cos], axis=1)
        sequences_per_station[station_id] = features
        targets_per_station[station_id] = log_counts  # using same transform for targets
        dates_per_station[station_id] = group["trip_date"]

    return station_ids, station_to_idx, sequences_per_station, targets_per_station, dates_per_station


class StationSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: List[np.ndarray],
        station_indices: List[int],
        targets: List[float],
    ) -> None:
        assert len(sequences) == len(station_indices) == len(targets)
        self._sequences = [torch.from_numpy(seq.astype(np.float32)) for seq in sequences]
        self._stations = torch.tensor(station_indices, dtype=torch.long)
        self._targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._sequences[idx], self._stations[idx], self._targets[idx]


class StationDemandGRU(nn.Module):
    def __init__(
        self,
        num_stations: int,
        feature_dim: int,
        hidden_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_stations, embedding_dim)
        self.gru = nn.GRU(feature_dim + embedding_dim, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor, station_idx: torch.Tensor) -> torch.Tensor:
        # sequences: (batch, seq_len, feature_dim)
        emb = self.embedding(station_idx)
        emb = emb.unsqueeze(1).expand(-1, sequences.size(1), -1)
        x = torch.cat([sequences, emb], dim=-1)
        outputs, _ = self.gru(x)
        last_output = outputs[:, -1, :]
        preds = self.projection(last_output)
        return preds.squeeze(-1)


def make_datasets(
    station_ids: Sequence[str],
    station_to_idx: Dict[str, int],
    sequences: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    dates: Dict[str, pd.Series],
    history_days: int,
    forecast_horizon: int,
    eval_days: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    samples_train: List[np.ndarray] = []
    stations_train: List[int] = []
    targets_train: List[float] = []

    samples_val: List[np.ndarray] = []
    stations_val: List[int] = []
    targets_val: List[float] = []

    all_dates = pd.concat(list(dates.values()))
    cutoff_date = all_dates.max() - pd.Timedelta(days=eval_days)

    for station_id in station_ids:
        feature_array = sequences[station_id]
        target_array = targets[station_id]
        date_series = dates[station_id].reset_index(drop=True)
        total = len(target_array)
        limit = total - forecast_horizon + 1
        if limit <= history_days:
            continue
        station_index = station_to_idx[station_id]
        for end_idx in range(history_days, limit):
            seq = feature_array[end_idx - history_days : end_idx]
            target_pos = end_idx + forecast_horizon - 1
            target_value = target_array[target_pos]
            target_date = date_series.iloc[target_pos]
            if target_date > cutoff_date:
                samples_val.append(seq)
                stations_val.append(station_index)
                targets_val.append(float(target_value))
            else:
                samples_train.append(seq)
                stations_train.append(station_index)
                targets_train.append(float(target_value))

    if not samples_train:
        raise RuntimeError("Training split ended up empty; adjust history or data range")

    train_ds = StationSequenceDataset(samples_train, stations_train, targets_train)
    val_ds: Optional[Dataset] = None
    if samples_val:
        val_ds = StationSequenceDataset(samples_val, stations_val, targets_val)
    return train_ds, val_ds


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for sequences, station_idx, targets in loader:
        sequences = sequences.to(device)
        station_idx = station_idx.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(sequences, station_idx)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        batch_size = sequences.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    if loader is None or len(loader.dataset) == 0:
        return {"loss": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    model.eval()
    total_loss = 0.0
    total_examples = 0
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    with torch.no_grad():
        for sequences, station_idx, targets in loader:
            sequences = sequences.to(device)
            station_idx = station_idx.to(device)
            targets = targets.to(device)

            preds = model(sequences, station_idx)
            loss = criterion(preds, targets)

            batch_size = sequences.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds_concat = np.concatenate(preds_all)
    targets_concat = np.concatenate(targets_all)
    preds_counts = np.expm1(preds_concat)
    target_counts = np.expm1(targets_concat)
    rmse = float(np.sqrt(np.mean((preds_counts - target_counts) ** 2)))
    mae = float(np.mean(np.abs(preds_counts - target_counts)))

    return {
        "loss": total_loss / max(total_examples, 1),
        "rmse": rmse,
        "mae": mae,
    }


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    device = choose_device()
    logging.info("Using device: %s", device)

    csv_paths = find_csv_files(args.data_root)
    trips = load_all_trips(csv_paths)

    daily = aggregate_daily_counts(trips)
    daily = filter_min_history(daily, args.min_station_days)
    if daily.empty:
        raise RuntimeError("No stations left after filtering; try lowering --min-station-days")
    expanded = expand_station_timeseries(daily)

    station_ids, station_to_idx, sequences, targets, dates = build_feature_arrays(expanded)
    train_ds, val_ds = make_datasets(
        station_ids,
        station_to_idx,
        sequences,
        targets,
        dates,
        history_days=args.history_days,
        forecast_horizon=args.forecast_horizon,
        eval_days=args.eval_days,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    model = StationDemandGRU(
        num_stations=len(station_ids),
        feature_dim=next(iter(sequences.values())).shape[1],
        hidden_size=args.hidden_size,
        embedding_dim=args.embedding_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)
        logging.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%s val_rmse=%s val_mae=%s",
            epoch,
            args.epochs,
            train_loss,
            _format_metric(metrics["loss"]),
            _format_metric(metrics["rmse"]),
            _format_metric(metrics["mae"]),
        )
        if not math.isnan(metrics["loss"]) and metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            best_state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        logging.warning("Validation split empty; saving last model state")
        best_state = {"model_state": model.state_dict(), "epoch": args.epochs, "val_loss": None}

    artifact_path = args.output_dir / "station_demand_gru.pt"
    torch.save(
        {
            "model_state_dict": best_state["model_state"],
            "station_to_idx": station_to_idx,
            "config": {
                "history_days": args.history_days,
                "forecast_horizon": args.forecast_horizon,
                "hidden_size": args.hidden_size,
                "embedding_dim": args.embedding_dim,
                "feature_dim": next(iter(sequences.values())).shape[1],
                "num_stations": len(station_ids),
                "eval_days": args.eval_days,
                "seed": args.seed,
            },
        },
        artifact_path,
    )
    logging.info("Saved model to %s", artifact_path)

    final_metrics = evaluate(model, val_loader, criterion, device)
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w") as fp:
        json.dump(
            {
                "val_loss": final_metrics["loss"],
                "val_rmse": final_metrics["rmse"],
                "val_mae": final_metrics["mae"],
            },
            fp,
            indent=2,
        )
    logging.info("Saved metrics to %s", metrics_path)


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
