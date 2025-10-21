#!/usr/bin/env python3
"""
Job 4: Raster counts extraction — Delta-only, functional (no classes)

Behavior:
- Expects GRID_SOURCE to be a Delta table spec (catalog.schema.table) containing the full grid rows.
  Required columns in that table: grid_id, tile_id, grid_minx, grid_miny, grid_maxx, grid_maxy
  Optional columns supported: is_boundary, centroid_x, centroid_y, lon, lat, i_idx, j_idx
- The script iterates tile-by-tile: it pulls only that tile's rows from the Delta table (Spark -> pandas)
  and processes windows against local/extracted rasters under BUILT_ROOT/{tile_id} and SMOD_ROOT/{tile_id}.
- Uses ThreadPoolExecutor per tile to process chunks of grid rows (chunk_size). Optionally stages rasters
  to local SSD to speed many small window reads.
- Writes per-tile CSVs (optional) and a combined CSV; optionally writes combined results to a Delta table.
- No CSV fallback for grid input: GRID_SOURCE must be a Delta table.

Parameters (required):
- grid_source         : Delta table spec containing grid rows (e.g. prp_mr_bdap_projects.geospatialsolutions.grid_centroids)
- built_root          : filesystem path root for built tiles (contains per-tile subfolders)
- smod_root           : filesystem path root for smod tiles (contains per-tile subfolders)
- output_dir          : filesystem path to write CSV outputs
- counts_delta_table  : Delta table spec to write combined counts (required if you want Delta output)

Optional (have sensible defaults but must be provided as parameters to validate):
- use_smod            : True/False
- include_nodata      : True/False
- add_percentages     : True/False
- use_boundary_mask   : True/False
- admin_path, admin_field, admin_value, tile_footprint_path, tile_id_field, target_crs
- chunk_size, max_workers, stage_to_local, local_dir, save_per_tile, write_mode

How to pass parameters when using Spark Python Task:
- Provide a JSON array of strings in the Task "Parameters" field (Databricks expects JSON array).
- This script contains a small wrapper that maps "--key value" pairs to environment variables (UPPERCASE),
  so use flags like "--grid_source", "--built_root", etc.
- Example JSON arrays are provided at the end of this file (dry-run and real run examples).

Notes / safety:
- This implementation pulls only per-tile rows to pandas, avoiding toPandas() for the entire table.
- If you expect extremely large numbers of rows per tile, consider a distributed Spark approach (mapPartitions).
- Ensure raster/geospatial libraries are available on the cluster (init script / task libraries).

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
import tempfile
import shutil
import traceback
from typing import Optional, Sequence, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.ops import unary_union
import psutil

from pyspark.sql import SparkSession

# ---------------------------
# Widget / env helpers
# ---------------------------
def get_widget_optional(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return dbutils.widgets.get(name)  # type: ignore
    except Exception:
        return os.environ.get(name.upper(), default)

def get_widget_required(name: str) -> str:
    val = get_widget_optional(name, None)
    if val is None or str(val).strip() == "":
        example_json = (
            '["--grid_source","prp_mr_bdap_projects.geospatialsolutions.grid_centroids",'
            '"--built_root","/mnt/data/tiles/built_c","--smod_root","/mnt/data/tiles/smod",'
            '"--output_dir","/mnt/data/outputs","--counts_delta_table","prp_mr_bdap_projects.geospatialsolutions.counts_combined"]'
        )
        raise RuntimeError(
            f"Required job parameter '{name}' is missing.\n\n"
            f"Provide parameters as a JSON array in the Job task parameters or as notebook parameters.\n"
            f"Example JSON array:\n  {example_json}\n"
        )
    return str(val)

# ---------------------------
# Required job parameters (no defaults)
# ---------------------------
GRID_SOURCE = get_widget_required("grid_source")
BUILT_ROOT = get_widget_required("built_root")
SMOD_ROOT = get_widget_required("smod_root")
OUTPUT_DIR = get_widget_required("output_dir")
COUNTS_DELTA_TABLE = get_widget_required("counts_delta_table")

# Optional / defaults
USE_SMOD = get_widget_optional("use_smod", "True").lower() in ("true","1","t","yes")
INCLUDE_NODATA = get_widget_optional("include_nodata", "False").lower() in ("true","1","t","yes")
ADD_PERCENTAGES = get_widget_optional("add_percentages", "True").lower() in ("true","1","t","yes")
USE_BOUNDARY_MASK = get_widget_optional("use_boundary_mask", "False").lower() in ("true","1","t","yes")

ADMIN_PATH = get_widget_optional("admin_path", None)
ADMIN_FIELD = get_widget_optional("admin_field", "ISO3")
ADMIN_VALUE = get_widget_optional("admin_value", "IND")
TILE_FOOTPRINT_PATH = get_widget_optional("tile_footprint_path", None)
TILE_ID_FIELD = get_widget_optional("tile_id_field", "tile_id")
TARGET_CRS = get_widget_optional("target_crs", "ESRI:54009")

CHUNK_SIZE = int(get_widget_optional("chunk_size", "5000"))
MAX_WORKERS = int(get_widget_optional("max_workers", "8"))
STAGE_TO_LOCAL = get_widget_optional("stage_to_local", "True").lower() in ("true","1","t","yes")
LOCAL_DIR = get_widget_optional("local_dir", "/local_disk0/raster_cache")
SAVE_PER_TILE = get_widget_optional("save_per_tile", "False").lower() in ("true","1","t","yes")
WRITE_MODE = get_widget_optional("write_mode", "overwrite")

# Defensive prints to confirm parameters
print("JOB PARAMETERS:")
print(" GRID_SOURCE         =", GRID_SOURCE)
print(" BUILT_ROOT          =", BUILT_ROOT)
print(" SMOD_ROOT           =", SMOD_ROOT)
print(" OUTPUT_DIR          =", OUTPUT_DIR)
print(" COUNTS_DELTA_TABLE  =", COUNTS_DELTA_TABLE)
print(" USE_SMOD            =", USE_SMOD)
print(" INCLUDE_NODATA      =", INCLUDE_NODATA)
print(" ADD_PERCENTAGES     =", ADD_PERCENTAGES)
print(" USE_BOUNDARY_MASK   =", USE_BOUNDARY_MASK)
print(" ADMIN_PATH          =", ADMIN_PATH)
print(" ADMIN_FIELD         =", ADMIN_FIELD)
print(" ADMIN_VALUE         =", ADMIN_VALUE)
print(" TILE_FOOTPRINT_PATH =", TILE_FOOTPRINT_PATH)
print(" CHUNK_SIZE          =", CHUNK_SIZE)
print(" MAX_WORKERS         =", MAX_WORKERS)
print(" STAGE_TO_LOCAL      =", STAGE_TO_LOCAL)
print(" LOCAL_DIR           =", LOCAL_DIR)
print(" SAVE_PER_TILE       =", SAVE_PER_TILE)
print(" WRITE_MODE          =", WRITE_MODE)

# ---------------------------
# Constants
# ---------------------------
BUILT_CLASSES = np.array([11,12,13,14,15,21,22,23,24,25], dtype=np.uint8)
SMOD_CLASSES  = np.array([0,10,11,12,13,21,22,23,30], dtype=np.uint8)
SMOD_RECLASS_MAP = {30:2, 23:2, 22:1, 21:1, 13:0, 12:0, 11:0, 10:0, 0:0}
NODATA = 255

# ---------------------------
# Low-level helpers
# ---------------------------
def _bincount_classes(flat: np.ndarray, classes: np.ndarray) -> np.ndarray:
    bc = np.bincount(flat, minlength=256)
    return bc[classes].astype(np.int32)

def _count_window(arr: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, int, int]:
    flat = arr.ravel().astype(np.uint8, copy=False)
    nodata_mask = (flat == NODATA)
    if nodata_mask.all():
        return np.zeros(len(classes), dtype=np.int32), 0, flat.size
    valid = flat[~nodata_mask]
    if valid.size == 0:
        return np.zeros(len(classes), dtype=np.int32), 0, int(nodata_mask.sum())
    counts = _bincount_classes(valid, classes)
    return counts, valid.size, int(nodata_mask.sum())

def _mask_boundary_window(arr, window, src_transform, clip_geom):
    h, w = arr.shape
    win_transform = rasterio.windows.transform(window, src_transform)
    mask = rasterize([(clip_geom, 1)], out_shape=(h, w), transform=win_transform, fill=0, dtype='uint8')
    if mask.shape == arr.shape:
        arr[mask == 0] = NODATA
    else:
        mh, mw = mask.shape
        h0, w0 = min(h, mh), min(w, mw)
        arr[:h0, :w0][mask[:h0, :w0] == 0] = NODATA

def find_raster_for_tile(tile_id: str, root: str) -> Optional[str]:
    d = os.path.join(root, str(tile_id))
    if not os.path.isdir(d):
        return None
    tifs = [f for f in os.listdir(d) if f.lower().endswith(".tif")]
    if not tifs:
        return None
    return os.path.join(d, tifs[0])

def stage_to_local_if_needed(src_path: Optional[str], local_dir: str, stage: bool, verbose: bool) -> Optional[str]:
    if not stage or not src_path or not os.path.exists(src_path):
        return src_path
    os.makedirs(local_dir, exist_ok=True)
    dst = os.path.join(local_dir, os.path.basename(src_path))
    try:
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src_path):
            if verbose:
                print(f"Using cached staged file: {dst}")
            return dst
        shutil.copyfile(src_path, dst)
        if verbose:
            print(f"Staged raster to local: {dst} ({os.path.getsize(src_path)/1e6:.1f} MB)")
        return dst
    except Exception as e:
        if verbose:
            print(f"WARN: staging failed ({e}), falling back to original path")
        return src_path

# ---------------------------
# Chunk processor (reads windows)
# ---------------------------
def _process_chunk(raster_path: str,
                   chunk_df: pd.DataFrame,
                   dataset: str,
                   class_vec: np.ndarray,
                   use_boundary_mask: bool,
                   tile_clip: Optional[Any],
                   pass_through_columns: List[str],
                   include_nodata: bool,
                   add_percentages: bool,
                   verbose: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with rasterio.open(raster_path) as src:
        for _, row in chunk_df.iterrows():
            try:
                w = from_bounds(row.grid_minx, row.grid_miny, row.grid_maxx, row.grid_maxy, src.transform)
                arr = src.read(1, window=w, boundless=False)
            except Exception as e:
                if verbose:
                    print(f"  ERROR reading window for grid {row.get('grid_id')} : {e}")
                continue

            if use_boundary_mask and tile_clip is not None and int(row.get('is_boundary', 0)) == 1:
                _mask_boundary_window(arr, w, src.transform, tile_clip)

            counts_vec, valid_count, nodata_count = _count_window(arr, class_vec)
            h, w_actual = arr.shape
            rec: Dict[str, Any] = {
                "grid_id": row.grid_id,
                "tile_id": row.tile_id,
                "is_boundary": int(row.is_boundary) if 'is_boundary' in row else 0,
                f"{dataset}_total_valid_pixels": int(valid_count),
                f"{dataset}_actual_window_pixels": int(h * w_actual),
            }
            for col in pass_through_columns:
                if col in row:
                    rec[col] = row[col]
            if include_nodata:
                rec[f"{dataset}_nodata_pixels"] = int(nodata_count)
            for ccode, ccnt in zip(class_vec, counts_vec):
                rec[f"{dataset}_class_{int(ccode)}"] = int(ccnt)
            if add_percentages and valid_count > 0:
                for ccode, ccnt in zip(class_vec, counts_vec):
                    rec[f"{dataset}_pct_class_{int(ccode)}"] = float(ccnt) / float(valid_count)
            if dataset == "smod":
                r0 = r1 = r2 = 0
                for ccode, ccnt in zip(class_vec, counts_vec):
                    grp = SMOD_RECLASS_MAP.get(int(ccode), None)
                    if grp == 0: r0 += int(ccnt)
                    elif grp == 1: r1 += int(ccnt)
                    elif grp == 2: r2 += int(ccnt)
                rec["smod_reclass_0"] = int(r0)
                rec["smod_reclass_1"] = int(r1)
                rec["smod_reclass_2"] = int(r2)
            out.append(rec)
    return out

# ---------------------------
# Per-tile processing (Spark -> pandas per tile)
# ---------------------------
def process_tile(tile_id: str,
                 spark: SparkSession,
                 grid_table: str,
                 dataset: str,
                 built_root: str,
                 smod_root: str,
                 chunk_size: int,
                 max_workers: int,
                 use_boundary_mask: bool,
                 tile_clip_map: Dict[str, Any],
                 pass_through_columns: List[str],
                 include_nodata: bool,
                 add_percentages: bool,
                 stage_to_local: bool,
                 local_dir: str,
                 verbose: bool) -> Optional[pd.DataFrame]:

    # Read only rows for this tile from the Delta table into pandas
    try:
        sdf_tile = spark.read.table(grid_table).where(f"tile_id = '{tile_id}'")
        tile_rows = sdf_tile.toPandas()
    except Exception as e:
        if verbose:
            print(f"  ERROR reading grid rows for tile {tile_id}: {e}")
        return None

    if tile_rows.empty:
        if verbose:
            print(f"  No grid rows for tile {tile_id}")
        return None

    raster_root = built_root if dataset == "built_c" else smod_root
    raster_path = find_raster_for_tile(tile_id, raster_root)
    if raster_path is None:
        if verbose:
            print(f"  [{dataset}] No raster found for tile {tile_id} under {raster_root}")
        return None

    raster_path = stage_to_local_if_needed(raster_path, local_dir, stage_to_local, verbose)
    class_vec = BUILT_CLASSES if dataset == "built_c" else SMOD_CLASSES
    mask_this = use_boundary_mask and ((dataset == "built_c") or (dataset == "smod" and False))
    tile_clip = tile_clip_map.get(tile_id) if mask_this else None

    chunks = [tile_rows.iloc[i:i+chunk_size] for i in range(0, len(tile_rows), chunk_size)]
    if verbose:
        print(f"  [{dataset}] windows={len(tile_rows)} chunks={len(chunks)} mask={mask_this} path={'local' if raster_path and raster_path.startswith(local_dir) else 'remote'}")

    results: List[Dict[str, Any]] = []
    def worker(ch_df):
        try:
            return _process_chunk(raster_path, ch_df, dataset, class_vec, use_boundary_mask, tile_clip, pass_through_columns, include_nodata, add_percentages, verbose)
        except Exception as e:
            if verbose:
                print(f"    Worker error: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, ch) for ch in chunks]
        done = 0
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.extend(res)
            done += 1
            if verbose:
                pct = done / max(1, len(chunks)) * 100
                if done % max(1, len(chunks)//5) == 0 or done == len(chunks):
                    print(f"    [{dataset}] chunk progress {done}/{len(chunks)} ({pct:.1f}%)")

    if not results:
        return None
    return pd.DataFrame(results)

# ---------------------------
# Boundary clip preparation
# ---------------------------
def prepare_tile_clips(grid_table: str, spark: SparkSession, admin_path: str, admin_field: str, admin_value: str, tile_footprint_path: str, tile_id_field: str, target_crs: str, verbose: bool) -> Dict[str, Any]:
    if not admin_path or not tile_footprint_path:
        raise RuntimeError("admin_path and tile_footprint_path are required for boundary masking")
    admin = gpd.read_file(admin_path)
    if admin_field not in admin.columns:
        raise RuntimeError(f"Admin field {admin_field} not in admin columns")
    sel = admin[admin[admin_field] == admin_value].to_crs(target_crs)
    if sel.empty:
        raise RuntimeError(f"No admin rows match {admin_field}={admin_value}")
    union_geom = unary_union(sel.geometry)
    tiles = gpd.read_file(tile_footprint_path).to_crs(target_crs)
    grid_tile_ids = spark.read.table(grid_table).select("tile_id").distinct().rdd.map(lambda r: r[0]).collect()
    grid_tile_set = set(grid_tile_ids)
    tile_clip_map: Dict[str, Any] = {}
    for _, tr in tiles.iterrows():
        tid = tr[tile_id_field]
        if tid not in grid_tile_set:
            continue
        geom = tr.geometry
        if geom is None or not geom.intersects(union_geom):
            continue
        clip = geom.intersection(union_geom)
        if not clip.is_empty:
            tile_clip_map[tid] = clip
    if verbose:
        print(f"Prepared boundary clips for {len(tile_clip_map)} tiles")
    return tile_clip_map

# ---------------------------
# Main workflow
# ---------------------------
def main():
    start_time = time.time()
    spark = SparkSession.builder.getOrCreate()

    # Validate grid_source is a table spec (enforce delta-only)
    if os.path.sep in GRID_SOURCE:
        raise RuntimeError("GRID_SOURCE appears to be a filesystem path; this job requires a Delta table spec (catalog.schema.table).")

    # Get distinct tile ids from the Delta table
    print(f"Listing distinct tile IDs from table: {GRID_SOURCE}")
    tile_ids = spark.read.table(GRID_SOURCE).select("tile_id").distinct().rdd.map(lambda r: r[0]).collect()
    tile_ids = [str(t) for t in tile_ids if t is not None and str(t).strip() != ""]
    if not tile_ids:
        raise RuntimeError(f"No tile_ids found in table {GRID_SOURCE}")
    print(f"Found {len(tile_ids)} unique tiles")

    # Optional prepare tile clips if using boundary mask
    tile_clip_map: Dict[str, Any] = {}
    if USE_BOUNDARY_MASK:
        tile_clip_map = prepare_tile_clips(GRID_SOURCE, spark, ADMIN_PATH, ADMIN_FIELD, ADMIN_VALUE, TILE_FOOTPRINT_PATH, TILE_ID_FIELD, TARGET_CRS, verbose=True)

    pass_through_columns = ["centroid_x","centroid_y","lon","lat","i_idx","j_idx"]
    outputs: List[pd.DataFrame] = []

    print(f"Processing {len(tile_ids)} tiles | workers={MAX_WORKERS} chunk_size={CHUNK_SIZE} stage_to_local={STAGE_TO_LOCAL}")

    for idx, tile in enumerate(sorted(tile_ids), start=1):
        print(f"\n=== Tile {idx}/{len(tile_ids)}: {tile} ===")
        tile_frames: List[pd.DataFrame] = []

        # built_c
        if os.path.isdir(os.path.join(BUILT_ROOT, str(tile))):
            df_built = process_tile(tile, spark, GRID_SOURCE, "built_c", BUILT_ROOT, SMOD_ROOT, CHUNK_SIZE, MAX_WORKERS, USE_BOUNDARY_MASK, tile_clip_map, pass_through_columns, INCLUDE_NODATA, ADD_PERCENTAGES, STAGE_TO_LOCAL, LOCAL_DIR, True)
            if df_built is not None:
                tile_frames.append(df_built)
        else:
            print(f"  built_c folder not found for tile {tile} under {BUILT_ROOT}")

        # smod
        if USE_SMOD:
            if os.path.isdir(os.path.join(SMOD_ROOT, str(tile))):
                df_smod = process_tile(tile, spark, GRID_SOURCE, "smod", BUILT_ROOT, SMOD_ROOT, CHUNK_SIZE, MAX_WORKERS, USE_BOUNDARY_MASK, tile_clip_map, pass_through_columns, INCLUDE_NODATA, ADD_PERCENTAGES, STAGE_TO_LOCAL, LOCAL_DIR, True)
                if df_smod is not None:
                    tile_frames.append(df_smod)
            else:
                print(f"  smod folder not found for tile {tile} under {SMOD_ROOT}")

        if not tile_frames:
            print(f"  No data produced for tile {tile}")
            continue

        # Merge dataset frames for tile
        merged = tile_frames[0]
        for extra in tile_frames[1:]:
            merge_on = ["grid_id","tile_id","is_boundary"] + [c for c in pass_through_columns if c in extra.columns]
            merged = merged.merge(extra, on=merge_on, how="outer")

        if SAVE_PER_TILE:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_tile = os.path.join(OUTPUT_DIR, f"class_counts_{tile}_{ts}.csv")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            merged.to_csv(out_tile, index=False)
            print(f"  Saved per-tile CSV: {out_tile}")

        outputs.append(merged)

    if not outputs:
        print("No tiles produced data. Exiting.")
        return

    combined = pd.concat(outputs, ignore_index=True)
    ts_all = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_all = os.path.join(OUTPUT_DIR, f"class_counts_all_{ts_all}.csv")
    combined.to_csv(out_all, index=False)
    print(f"\nSaved combined counts CSV: {out_all} rows={len(combined):,}")

    # Save to Delta table (attempt)
    try:
        print(f"Attempting to write combined counts to Delta table: {COUNTS_DELTA_TABLE} (mode={WRITE_MODE})")
        sdf = spark.createDataFrame(combined)
        sdf.write.format("delta").mode(WRITE_MODE).saveAsTable(COUNTS_DELTA_TABLE)
        print("Saved combined counts to Delta.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Warning: failed to write combined counts to Delta: {e}\n{tb}")

    elapsed = time.time() - start_time
    print(f"Job completed in {elapsed:.1f}s")

if __name__ == "__main__":
    main()