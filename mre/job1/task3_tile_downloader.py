#!/usr/bin/env python3
"""
Task 3 — Download GHSL Tiles from JRC Repository

Downloads GHSL (Global Human Settlement Layer) raster tiles for grid cells
generated in Task 2. Downloads zip files from JRC FTP, extracts TIFFs.

Configuration:
--------------
Reads from config.json (generated from config.yaml via config_builder.py).
All paths are auto-generated from the YAML configuration.

Required config keys:
  - grid_source: Delta table with grid centroids (contains tile_id column)
  - tiles_dest_root: Root directory for downloaded tiles
  - download_status_table: Delta table to track download status
  - datasets: Comma-separated datasets to download (e.g., "built_c,smod")
  - download_concurrency: Number of parallel downloads (default: 3)
  - download_retries: Number of retry attempts (default: 2)

Usage:
------
  python task3_tile_downloader.py --config_path config.json

Or with CLI overrides:
  python task3_tile_downloader.py --config_path config.json --dry_run True

Output:
-------
  - Downloaded tiles: {tiles_dest_root}/built_c/{tile_id}/*.tif
  - Downloaded tiles: {tiles_dest_root}/smod/{tile_id}/*.tif
  - Delta table: {catalog}.{schema}.download_status (tracks success/failure)

Datasets:
---------
  - built_c: Building construction layer (10m resolution)
  - smod: Settlement model layer (1km resolution)

Notes:
------
  - Uses concurrent ThreadPoolExecutor for parallel downloads
  - Automatically retries failed downloads
  - Verifies extracted files exist
  - Compatible with Spark Connect (no .rdd usage)
"""
import sys
import os
import time
import zipfile
import tempfile
import shutil
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import requests
import pandas as pd
from pyspark.sql import SparkSession

import requests
import pandas as pd
from pyspark.sql import SparkSession

# os.environ["CONFIG_PATH"] = "./config.json"


# ---- CLI -> ENV wrapper (keeps compatibility with Databricks job task parameters) ----
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        key = args[i]
        if key.startswith("--"):
            env_key = key.lstrip("-").upper()
            value = ""
            if (i + 1) < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                i += 1
            if value != "":
                os.environ[env_key] = value
        else:
            i += 1
# -------------------------------------------------------------------------------------

# ---------------------------
# GHSL config (as provided)
# ---------------------------
GHSL_CONFIGS = {
    "built_c": {
        "base_url": (
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_BUILT_C_GLOBE_R2023A/"
            "GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10/V1-0/tiles/"
        ),
        "template": "GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0_{tile_id}.zip"
    },
    "smod": {
        "base_url": (
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_SMOD_GLOBE_R2023A/"
            "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V2-0/tiles/"
        ),
        "template": "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V2_0_{tile_id}.zip"
    }
}

# ---------------------------
# Config loader (JSON or env overrides)
# ---------------------------
DEFAULT_CONFIG = {
    "grid_source": None,
    "tiles_dest_root": None,
    "download_status_table": None,
    "datasets": "built_c,smod",
    "download_concurrency": 3,
    "download_retries": 2,
    "dry_run": True,
    "spark_tmp_dir": "/tmp/job3_grid_tmp",
    "write_mode": "overwrite"
}

def _read_json_path(path: str) -> dict:
    if path.startswith("dbfs:"):
        local = path.replace("dbfs:", "/dbfs", 1)
    else:
        local = path
    with open(local, "r") as fh:
        import json
        return json.load(fh)

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG", "") or os.environ.get("CONFIG_JSON", "")
    if cfg_path:
        try:
            loaded = _read_json_path(cfg_path)
            if not isinstance(loaded, dict):
                raise RuntimeError("config JSON must be an object")
            cfg.update(loaded)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{cfg_path}': {e}")
    # apply individual env overrides (CLI wrapper sets these)
    for k in list(cfg.keys()):
        ek = k.upper()
        if ek in os.environ and os.environ[ek] != "":
            val = os.environ[ek]
            if isinstance(cfg[k], bool):
                cfg[k] = str(val).lower() in ("true","1","t","yes")
            elif isinstance(cfg[k], int):
                try:
                    cfg[k] = int(val)
                except Exception:
                    pass
            else:
                cfg[k] = val
    # minimal validation
    missing = [k for k in ("grid_source","tiles_dest_root","download_status_table") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {missing}")
    return cfg

def resolve_path(path: str) -> str:
    """Map dbfs:/... to /dbfs/... for local driver filesystem access."""
    if isinstance(path, str) and path.startswith("dbfs:"):
        return path.replace("dbfs:", "/dbfs", 1)
    return path


# ---------------------------
# Download helpers
# ---------------------------
def _download_one(tile_id: str,
                  dataset: str,
                  dest_root: str,
                  timeout: int = 300,
                  retries: int = 2,
                  dry_run: bool = False) -> Dict[str, Any]:
    config = GHSL_CONFIGS.get(dataset)
    if config is None:
        return {"tile_id": tile_id, "dataset": dataset, "status": "unsupported_dataset", "path": None, "url": None}
    extract_dir = Path(dest_root) / dataset / str(tile_id)
    extract_dir.mkdir(parents=True, exist_ok=True)
    existing_tifs = list(extract_dir.glob("*.tif"))
    if existing_tifs:
        return {"tile_id": tile_id, "dataset": dataset, "status": "exists", "path": str(existing_tifs[0]), "url": None}

    filename = config["template"].format(tile_id=tile_id)
    url = config["base_url"] + filename
    zip_path = extract_dir.parent / f"{tile_id}_{dataset}.zip"

    if dry_run:
        return {"tile_id": tile_id, "dataset": dataset, "status": "dry_run", "path": None, "url": url}

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # Extract zip
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
            try:
                zip_path.unlink(missing_ok=True)
            except Exception:
                pass
            tifs = list(extract_dir.glob("*.tif"))
            if not tifs:
                return {"tile_id": tile_id, "dataset": dataset, "status": "failed_no_tif", "path": None, "url": url}
            return {"tile_id": tile_id, "dataset": dataset, "status": "downloaded", "path": str(tifs[0]), "url": url}
        except Exception as e:
            if attempt == retries:
                return {"tile_id": tile_id, "dataset": dataset, "status": f"failed({e})", "path": None, "url": url}
            time.sleep(1 + attempt)
    return {"tile_id": tile_id, "dataset": dataset, "status": "failed_exhausted", "path": None, "url": url}

def download_tiles(tile_ids: List[str],
                   datasets: List[str],
                   dest_root: str,
                   concurrency: int = 3,
                   retries: int = 2,
                   dry_run: bool = False) -> List[Dict[str, Any]]:
    tasks = []
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for t in tile_ids:
            for ds in datasets:
                tasks.append(ex.submit(_download_one, t, ds, dest_root, 300, retries, dry_run))
        for fut in as_completed(tasks):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"tile_id": None, "dataset": None, "status": f"failed_exception({e})", "path": None, "url": None})
    return results

# ---------------------------
# Spark helper: collect distinct tile ids (Spark Connect safe)
# ---------------------------
def get_tile_ids_from_table(spark: SparkSession, table_spec: str) -> List[str]:
    """
    Return a sorted list of distinct tile_id values from a Delta table.
    This implementation avoids .rdd so it works with Spark Connect.
    """
    try:
        sdf = spark.read.table(table_spec)
    except Exception as e:
        raise RuntimeError(f"Could not read Spark table '{table_spec}': {e}")

    if "tile_id" not in sdf.columns:
        raise RuntimeError(f"Table '{table_spec}' does not contain required column 'tile_id'.")

    # Use collect() on the distinct selection (returns list[Row]) and extract values
    rows = sdf.select("tile_id").distinct().collect()
    tile_ids = []
    for r in rows:
        # Row may be a dict-like object; handle multiple possible row representations
        try:
            val = r["tile_id"]
        except Exception:
            # fallback to positional index
            try:
                val = r[0]
            except Exception:
                val = None
        if val is not None and str(val).strip() != "":
            tile_ids.append(str(val))
    return sorted(tile_ids)

# ---------------------------
# Main
# ---------------------------
def main():
    start = time.time()
    cfg = load_config()

    ISO3 = cfg.get("iso3", "IND").strip().upper()
    def add_iso_suffix(name, iso3=ISO3):
        if name is None:
            return name
        if name.upper().endswith(f"_{iso3}"):
            return name
        if name.endswith(".csv"):
            return name[:-4] + f"_{iso3}.csv"
        if name.endswith(".parquet"):
            return name[:-8] + f"_{iso3}.parquet"
        return f"{name}_{iso3}"

    GRID_SOURCE = add_iso_suffix(cfg["grid_source"])
    # Populate variables
    TILES_DEST_ROOT = cfg["tiles_dest_root"]
    #DOWNLOAD_STATUS_TABLE = cfg["download_status_table"]
    DOWNLOAD_STATUS_TABLE = add_iso_suffix(cfg["download_status_table"])
    DATASETS = [d.strip() for d in str(cfg.get("datasets", "built_c,smod")).split(",") if d.strip()]
    DOWNLOAD_CONCURRENCY = int(cfg.get("download_concurrency", 3))
    DOWNLOAD_RETRIES = int(cfg.get("download_retries", 2))
    DRY_RUN = bool(cfg.get("dry_run", True))
    SPARK_TMP_DIR = cfg.get("spark_tmp_dir", "/tmp/job3_grid_tmp")
    WRITE_MODE = cfg.get("write_mode", "overwrite")

    # Resolve dest root for driver filesystem (support dbfs:/)
    dest_root_local = resolve_path(TILES_DEST_ROOT)

    print("JOB PARAMETERS:")
    print(f" GRID_SOURCE            = {GRID_SOURCE}")
    print(f" TILES_DEST_ROOT        = {TILES_DEST_ROOT} -> resolved: {dest_root_local}")
    print(f" DOWNLOAD_STATUS_TABLE  = {DOWNLOAD_STATUS_TABLE}")
    print(f" DATASETS               = {DATASETS}")
    print(f" DOWNLOAD_CONCURRENCY   = {DOWNLOAD_CONCURRENCY}")
    print(f" DOWNLOAD_RETRIES       = {DOWNLOAD_RETRIES}")
    print(f" DRY_RUN                = {DRY_RUN}")
    print(f" SPARK_TMP_DIR          = {SPARK_TMP_DIR}")
    print(f" WRITE_MODE             = {WRITE_MODE}")

    spark = SparkSession.builder.getOrCreate()

    # Validate GRID_SOURCE looks like a table
    if os.path.sep in GRID_SOURCE:
        raise RuntimeError("GRID_SOURCE appears to be a filesystem path; this job requires a Delta table spec (catalog.schema.table).")

    # Get tile ids
    print(f"Reading distinct tile ids from table: {GRID_SOURCE}")
    tile_ids = get_tile_ids_from_table(spark, GRID_SOURCE)
    if not tile_ids:
        raise RuntimeError(f"No tile_ids found in table {GRID_SOURCE}")
    print(f"Found {len(tile_ids)} unique tiles (sample 10): {tile_ids[:10]}")

    if not DATASETS:
        raise RuntimeError("No datasets requested (provide datasets param, e.g. 'built_c,smod')")

    # Ensure local dest root exists
    try:
        Path(dest_root_local).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create dest root {dest_root_local}: {e}")

    print(f"Starting download step (dry_run={DRY_RUN})")
    statuses = download_tiles(tile_ids, DATASETS, dest_root_local, concurrency=DOWNLOAD_CONCURRENCY, retries=DOWNLOAD_RETRIES, dry_run=DRY_RUN)

    # Write status table to Delta via Spark
    try:
        sdf = spark.createDataFrame(pd.DataFrame(statuses))
        print(f"Writing download status to Delta table: {DOWNLOAD_STATUS_TABLE} (mode={WRITE_MODE})")
        writer = sdf.write.format("delta").mode(WRITE_MODE)
        if WRITE_MODE == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(DOWNLOAD_STATUS_TABLE)
        print("Download status written to Delta successfully.")
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to write download status to Delta table {DOWNLOAD_STATUS_TABLE}: {e}\n{tb}")

    elapsed = time.time() - start
    print(f"Download job completed in {elapsed:.1f}s -- {len(statuses)} status rows recorded.")

if __name__ == "__main__":
    main()
