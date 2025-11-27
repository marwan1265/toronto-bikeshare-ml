import logging
import json
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel
import sys
import os

# Add root directory to path so we can import prep_and_train
sys.path.append(str(Path(__file__).parent.parent))
from backend.fetch_stations import fetch_station_info
from backend.fetch_ridership import fetch_latest_data
from prep_and_train import (
    StationDemandGRU,
    load_all_trips,
    aggregate_daily_counts,
    filter_min_history,
    expand_station_timeseries,
    build_feature_arrays,
    find_csv_files
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Global state
model = None
station_to_idx = {}
idx_to_station = {}
station_metadata = {}
latest_sequences = {} # station_id -> tensor of shape (1, history_days, feature_dim)
config = {}
device = torch.device("cpu")

class StationResponse(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    capacity: Optional[int]
    predicted_demand: float

class HistoryPoint(BaseModel):
    date: str
    count: float

class ForecastResponse(BaseModel):
    station_id: str
    history: List[HistoryPoint]
    forecast: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, station_to_idx, idx_to_station, station_metadata, latest_sequences, config, device
    
    logger.info("Loading resources...")
    
    # 1. Fetch Station Info (GBFS)
    logger.info("Fetching station information...")
    fetch_station_info()
    
    # 2. Fetch Ridership Data (Open Data API)
    # We only fetch the most recent year of data to keep things fast and avoid using too much bandwidth
    logger.info("Fetching latest ridership data...")
    fetch_latest_data(latest_only=True)
    
    # 3. Load Station Metadata
    try:
        # Define paths for model and data (assuming they are in a 'model' and 'data' directory relative to the script)
        # Define paths for model and data
        # SCRIPT_DIR is .../backend, so parent is the project root
        ROOT_DIR = Path(__file__).parent.parent
        MODEL_PATH = ROOT_DIR / "artifacts" / "station_demand_gru.pt"
        STATION_INFO_PATH = ROOT_DIR / "data" / "station_information.json"

        if STATION_INFO_PATH.exists():
            with open(STATION_INFO_PATH, "r") as f:
                stations = json.load(f)
                station_metadata = {s["station_id"]: s for s in stations}
            logger.info(f"Loaded metadata for {len(station_metadata)} stations")
        else:
            logger.warning(f"Station info file not found at {STATION_INFO_PATH}")
    except Exception as e:
        logger.error(f"Failed to load station metadata: {e}")
        
    # 4. Load Model
    try:
        if MODEL_PATH.exists():
            logger.info(f"Loading model from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            config = checkpoint['config']
            station_to_idx = checkpoint['station_to_idx']
            idx_to_station = {v: k for k, v in station_to_idx.items()}
            
            model = StationDemandGRU(
                num_stations=config['num_stations'],
                feature_dim=config['feature_dim'],
                hidden_size=config['hidden_size'],
                embedding_dim=config['embedding_dim']
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model artifact not found at {MODEL_PATH}!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    # 5. Load Recent Data for Inference
    try:
        logger.info("Loading recent data for inference context...")
        # We'll try to use the history setting from the model config, but we'll fall back to a default if it's missing
        history_days = config.get("history_days", 14) 
        
        # Helper to load recent data
        def load_recent_data_helper(data_root: Path, history_days: int):
            csv_paths = find_csv_files(data_root)
            if not csv_paths:
                logger.warning("No CSV files found in data directory.")
                return {}
                
            trips = load_all_trips(csv_paths)
            daily = aggregate_daily_counts(trips)

            expanded = expand_station_timeseries(daily)
            
            # Build features
            _, _, sequences, _, _ = build_feature_arrays(expanded)
            
            # Extract last sequence for each station
            final_sequences = {}
            for sid, seq in sequences.items():
                if len(seq) >= history_days:
                    final_sequences[sid] = seq[-history_days:]
            return final_sequences

        raw_sequences = load_recent_data_helper(Path("data"), history_days=history_days)
        
        # Convert numpy sequences to tensors and add batch dimension
        for sid, seq_array in raw_sequences.items():
            tensor = torch.from_numpy(seq_array).float().unsqueeze(0).to(device)
            latest_sequences[sid] = tensor
            
        logger.info(f"Prepared inference context for {len(latest_sequences)} stations")
    except Exception as e:
        logger.error(f"Failed to load inference context: {e}")
    
    yield
    # Cleanup if needed
    pass

app = FastAPI(lifespan=lifespan, title="Toronto Bikeshare Forecast API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stations", response_model=List[StationResponse])
async def get_stations():
    results = []
    
    with torch.no_grad():
        for sid, meta in station_metadata.items():
            pred_value = 0.0
            
            # If we have a model and context for this station, predict
            if model and sid in latest_sequences and sid in station_to_idx:
                seq = latest_sequences[sid]
                s_idx = torch.tensor([station_to_idx[sid]]).to(device)
                
                # Forward pass
                pred_log = model(seq, s_idx)
                pred_count = float(np.expm1(pred_log.cpu().numpy())[0])
                pred_value = max(0.0, pred_count)
            
            results.append({
                "station_id": sid,
                "name": meta.get("name", "Unknown"),
                "lat": meta.get("lat", 0.0),
                "lon": meta.get("lon", 0.0),
                "capacity": meta.get("capacity"),
                "predicted_demand": pred_value
            })
            
    # Sort by demand desc
    results.sort(key=lambda x: x["predicted_demand"], reverse=True)
    return results

@app.get("/predict/{station_id}", response_model=ForecastResponse)
async def get_prediction(station_id: str):
    if station_id not in station_metadata:
        raise HTTPException(status_code=404, detail="Station not found")
        
    pred_value = 0.0
    history = []
    
    if model and station_id in latest_sequences and station_id in station_to_idx:
        with torch.no_grad():
            seq = latest_sequences[station_id]
            s_idx = torch.tensor([station_to_idx[station_id]]).to(device)
            pred_log = model(seq, s_idx)
            pred_value = float(np.expm1(pred_log.cpu().numpy())[0])
            pred_value = max(0.0, pred_value)
            
            # Reconstructing the history from the sequence.
            # Feature 0 is log(count + 1).
            seq_np = seq.squeeze(0).cpu().numpy()
            for i in range(len(seq_np)):
                log_cnt = seq_np[i, 0]
                cnt = float(np.expm1(log_cnt))
                history.append({
                    "date": f"Day -{len(seq_np)-i}", 
                    "count": max(0.0, cnt)
                })

    return {
        "station_id": station_id,
        "history": history,
        "forecast": pred_value
    }

@app.get("/health")
def health():
    return {"status": "ok", "stations_loaded": len(station_metadata), "model_loaded": model is not None}

# Serve Frontend
# Set up the assets folder so the frontend can load its JavaScript and CSS files
app.mount("/assets", StaticFiles(directory="web-ui/dist/assets"), name="assets")

# Catch-all route to serve index.html (for React Router)
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # If the request didn't match any API routes, we'll serve the frontend application.
    # First, check if the file exists in the dist folder (like favicon.ico).
    file_path = Path(f"web-ui/dist/{full_path}")
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
        
    # Otherwise return index.html
    return FileResponse("web-ui/dist/index.html")

