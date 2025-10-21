#!/usr/bin/env python3
"""
Job 3: Download GHSL tiles (zip -> extract TIFFs) if missing — Delta-only, no CSV fallback.

Behavior:
- Expects GRID_SOURCE to be a Delta table spec (catalog.schema.table) containing a column 'tile_id'.
- Extracts distinct tile_id values from that Delta table via Spark (no coalesce/CSV, no toPandas of the full table).
- Downloads GHSL zip tiles for requested datasets (built_c, smod) into DEST_ROOT/{dataset}/{tile_id}/ and extracts .tif files.
- Writes a download status table to a Delta table (DOWNLOAD_STATUS_TABLE). Overwrites by default.
- No CSV fallbacks — this script requires Delta inputs/outputs.

How to pass parameters to a Spark Python Task:
- Use the Databricks Task "Parameters" JSON array (e.g. ["--grid_source","schema.table", ...]).
- This script includes a small wrapper that maps --key value pairs to environment variables (UPPERCASE keys),
  so the rest of the code uses the same get_widget style as your other jobs.

Required parameters (no defaults):
- grid_source         : Delta table spec containing grid rows and tile_id column (e.g. prp_mr_bdap_projects.geospatialsolutions.grid_centroids)
- tiles_dest_root     : Filesystem path where tiles will be extracted (e.g. /Volumes/.../tiles)
- download_status_table: Delta table to write download statuses to (e.g. prp_mr_bdap_projects.geospatialsolutions.download_status)

Optional parameters:
- datasets            : comma-separated datasets to download (default: built_c,smod)
- download_concurrency: number of parallel download threads (default: 3)
- download_retries    : retries per tile (default: 2)
- dry_run             : True/False — if True, compute URLs and return statuses but do not download (default: False)
- spark_tmp_dir       : temporary dir for any spark-local work (default: /tmp/job3_grid_tmp)
- write_mode          : Delta write mode for the status table (default: overwrite)

Recommended flow:
1) Run with dry_run=True first to validate parameters and URLs.
2) Then run with dry_run=False to perform downloads.

Note: This script assumes the GHSL public URLs in GHSL_CONFIGS work from your environment (no auth).
If tiles are private / on S3, we can add boto3 support or signed-URL logic.
"""

# ---- Wrapper: map CLI JSON array params to env vars so widget-style code works ----
import sys
import os
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
# ----------------------------------------------------------------------------------

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

# GHSL config (kept as provided)
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
# Widget/env helpers
# ---------------------------
def get_widget_optional(name: str, default: Any = None) -> Any:
    """
    Try dbutils.widgets.get(name) (notebooks). If not available, read environment variable NAME (upper).
    """
    try:
        return dbutils.widgets.get(name)  # type: ignore
    except Exception:
        return os.environ.get(name.upper(), default)

def get_widget_required(name: str) -> str:
    """
    Fetch required param; raise a helpful error if missing.
    """
    val = get_widget_optional(name, None)
    if val is None or str(val).strip() == "":
        example_json = (
            '["--grid_source","prp_mr_bdap_projects.geospatialsolutions.grid_centroids",'
            '"--tiles_dest_root","/mnt/data/tiles","--download_status_table","prp_mr_bdap_projects.geospatialsolutions.download_status",'
            '"--datasets","built_c,smod","--download_concurrency","4","--download_retries","2","--dry_run","True"]'
        )
        raise RuntimeError(
            f"Required job parameter '{name}' is missing.\n\n"
            f"Provide parameters as a JSON array in the Job task parameters or as notebook parameters.\n"
            f"Example JSON array:\n  {example_json}\n"
        )
    return str(val)

# ---------------------------
# Required parameters (no defaults)
# ---------------------------
GRID_SOURCE = get_widget_required("grid_source")                      # Delta table spec
TILES_DEST_ROOT = get_widget_required("tiles_dest_root")             # e.g. /mnt/data/tiles
DOWNLOAD_STATUS_TABLE = get_widget_required("download_status_table") # Delta table to hold statuses

# Optional / with defaults
DATASETS = get_widget_optional("datasets", "built_c,smod").split(",")
DOWNLOAD_CONCURRENCY = int(get_widget_optional("download_concurrency", "3"))
DOWNLOAD_RETRIES = int(get_widget_optional("download_retries", "2"))
DRY_RUN = str(get_widget_optional("dry_run", "False")).lower() in ("true","1","t","yes")
SPARK_TMP_DIR = get_widget_optional("spark_tmp_dir", "/tmp/job3_grid_tmp")
WRITE_MODE = get_widget_optional("write_mode", "overwrite")  # Delta write mode for statuses

# Print out parameters upfront for validation
print("JOB PARAMETERS:")
print(f" GRID_SOURCE            = {GRID_SOURCE}")
print(f" TILES_DEST_ROOT        = {TILES_DEST_ROOT}")
print(f" DOWNLOAD_STATUS_TABLE  = {DOWNLOAD_STATUS_TABLE}")
print(f" DATASETS               = {DATASETS}")
print(f" DOWNLOAD_CONCURRENCY   = {DOWNLOAD_CONCURRENCY}")
print(f" DOWNLOAD_RETRIES       = {DOWNLOAD_RETRIES}")
print(f" DRY_RUN                = {DRY_RUN}")
print(f" SPARK_TMP_DIR          = {SPARK_TMP_DIR}")
print(f" WRITE_MODE             = {WRITE_MODE}")

# ---------------------------
# Download helper functions
# ---------------------------
def _download_one(tile_id: str,
                  dataset: str,
                  dest_root: str,
                  timeout: int = 300,
                  retries: int = 2,
                  dry_run: bool = False) -> Dict[str, Any]:
    config = GHSL_CONFIGS.get(dataset)
    if config is None:
        return {"tile_id": tile_id, "dataset": dataset, "status": "unsupported_dataset", "url": None}
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
    results = []
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
# Use Spark to extract distinct tile ids from Delta table (no CSV fallback)
# ---------------------------
def get_tile_ids_from_table(spark: SparkSession, table_spec: str) -> List[str]:
    """
    Return a sorted list of distinct tile_id values from a Delta table.
    """
    try:
        sdf = spark.read.table(table_spec)
    except Exception as e:
        raise RuntimeError(f"Could not read Spark table '{table_spec}': {e}")

    if "tile_id" not in sdf.columns:
        raise RuntimeError(f"Table '{table_spec}' does not contain required column 'tile_id'.")

    # distinct tile ids should be small; collect to driver
    tile_ids = sdf.select("tile_id").distinct().rdd.map(lambda r: r[0]).collect()
    tile_ids = [str(t) for t in tile_ids if t is not None and str(t).strip() != ""]
    return sorted(tile_ids)

# ---------------------------
# Main
# ---------------------------
def main():
    start = time.time()
    spark = SparkSession.builder.getOrCreate()

    # Validate GRID_SOURCE is treated as a table (no fallback)
    # If GRID_SOURCE contains a slash it's likely not a table; we enforce table-only here.
    if os.path.sep in GRID_SOURCE:
        raise RuntimeError("GRID_SOURCE appears to be a filesystem path; this job requires a Delta table spec (catalog.schema.table).")

    # Extract tile ids from the Delta table
    print(f"Reading distinct tile ids from table: {GRID_SOURCE}")
    tile_ids = get_tile_ids_from_table(spark, GRID_SOURCE)
    if not tile_ids:
        raise RuntimeError(f"No tile_ids found in table {GRID_SOURCE}")
    print(f"Found {len(tile_ids)} unique tiles")

    datasets = [d.strip() for d in DATASETS if d and str(d).strip() != ""]
    if not datasets:
        raise RuntimeError("No datasets requested (provide datasets param, e.g. 'built_c,smod')")

    print(f"Starting download check (dry_run={DRY_RUN}), datasets={datasets}, concurrency={DOWNLOAD_CONCURRENCY}")
    statuses = download_tiles(tile_ids, datasets, TILES_DEST_ROOT, concurrency=DOWNLOAD_CONCURRENCY, retries=DOWNLOAD_RETRIES, dry_run=DRY_RUN)

    # Write statuses to Delta table
    try:
        os.makedirs("/dbfs/tmp", exist_ok=True)  # ensure DBFS path exists for any intermediate (defensive)
    except Exception:
        pass

    try:
        # Convert to Spark DF and write as Delta
        sdf = spark.createDataFrame(pd.DataFrame(statuses))
        print(f"Writing download status to Delta table: {DOWNLOAD_STATUS_TABLE} (mode={WRITE_MODE})")
        sdf.write.format("delta").mode(WRITE_MODE).saveAsTable(DOWNLOAD_STATUS_TABLE)
        print("Download status written to Delta successfully.")
    except Exception as e:
        # Provide detailed message for troubleshooting
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to write download status to Delta table {DOWNLOAD_STATUS_TABLE}: {e}\n{tb}")

    elapsed = time.time() - start
    print(f"Download job completed in {elapsed:.1f}s -- {len(statuses)} status rows recorded.")

if __name__ == "__main__":
    main()