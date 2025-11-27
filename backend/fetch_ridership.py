import requests
import os
from pathlib import Path
import re

# Toronto Open Data CKAN API
BASE_URL = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
PACKAGE_ID = "bike-share-toronto-ridership-data"
DATA_DIR = Path("data")

def fetch_latest_data(latest_only: bool = False):
    print(f"Checking for new data from {BASE_URL}...")
    
    # 1. Get package metadata
    url = f"{BASE_URL}/api/3/action/package_show"
    params = {"id": PACKAGE_ID}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        package = response.json()
    except Exception as e:
        print(f"Failed to contact API: {e}")
        return

    if not package["success"]:
        print("API returned failure.")
        return

    resources = package["result"]["resources"]
    print(f"Found {len(resources)} resources available.")

    # 2. Iterate and download CSVs
    # We're looking for files that seem to be ridership CSVs.
    # They are usually named something like "Bike share ridership YYYY-MM".
    
    if not DATA_DIR.exists():
        os.makedirs(DATA_DIR)

    import zipfile
    import io

    # Let's gather all the valid resources first
    candidates = []
    for res in resources:
        name = res["name"]
        fmt = res["format"].lower()
        
        if "ridership" in name.lower() and (fmt == "zip" or fmt == "csv"):
            match = re.search(r'(\d{4})', name)
            if match:
                year = int(match.group(1))
                candidates.append({
                    "year": year,
                    "resource": res,
                    "name": name,
                    "fmt": fmt
                })
    
    # Sort them so the newest years come first
    candidates.sort(key=lambda x: x["year"], reverse=True)
    
    if latest_only and candidates:
        print(f"Latest only mode: selecting {candidates[0]['name']}")
        candidates = [candidates[0]]

    for item in candidates:
        name = item["name"]
        year = item["year"]
        res = item["resource"]
        fmt = item["fmt"]
        
        target_dir = DATA_DIR / f"bikeshare-ridership-{year}"
        
        if target_dir.exists():
            print(f"Skipping {name} (directory {target_dir} exists)")
            continue
            
        print(f"Downloading {name}...")
        try:
            download_url = res["url"]
            resp = requests.get(download_url)
            resp.raise_for_status()
            
            if fmt == "zip":
                print(f"Extracting to {target_dir}...")
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    z.extractall(target_dir)
            else:
                # It looks like a CSV file
                if not target_dir.exists():
                    os.makedirs(target_dir)
                filename = f"{name}.csv"
                with open(target_dir / filename, "wb") as f:
                    f.write(resp.content)
                    
            print(f"Successfully processed {name}")
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")

if __name__ == "__main__":
    fetch_latest_data()
