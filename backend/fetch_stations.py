import requests
import json
import os
from pathlib import Path

GBFS_URL = "https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information"
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "station_information.json"

def fetch_station_info():
    print(f"Fetching station data from {GBFS_URL}...")
    try:
        response = requests.get(GBFS_URL)
        response.raise_for_status()
        data = response.json()
        
        stations = []
        if "data" in data and "stations" in data["data"]:
            for s in data["data"]["stations"]:
                stations.append({
                    "station_id": s.get("station_id"),
                    "name": s.get("name"),
                    "lat": s.get("lat"),
                    "lon": s.get("lon"),
                    "capacity": s.get("capacity")
                })
            
            # Save to file
            with open(OUTPUT_FILE, "w") as f:
                json.dump(stations, f, indent=2)
            
            print(f"Successfully saved {len(stations)} stations to {OUTPUT_FILE}")
            return stations
        else:
            print("Error: Unexpected JSON structure")
            return []
            
    except Exception as e:
        print(f"Failed to fetch stations: {e}")
        return []

if __name__ == "__main__":
    if not DATA_DIR.exists():
        os.makedirs(DATA_DIR)
    fetch_station_info()
