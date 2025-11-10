#!/usr/bin/env python3
"""
Task 4 — Extract Raster Statistics from GHSL Tiles

Extracts building counts from GHSL raster tiles at grid centroid locations.
Uses parallel processing for optimal performance.

Configuration:
--------------
Reads from config.json (generated from config.yaml via config_builder.py).
All paths and table names are auto-generated from the YAML configuration.

Required config keys:
  - grid_source: Delta table with grid centroids
  - built_root: Path to built_c tiles directory
  - smod_root: Path to SMOD tiles directory (optional)
  - counts_delta_table: Output Delta table for counts
  - admin_path: Admin boundaries for masking
  - tile_footprint_path: Tile footprint shapefile
  - use_smod: Whether to include SMOD urban classification (default: True)
  - chunk_size: Processing chunk size (default: 10000)
  - max_workers: Thread pool size for chunk processing (default: 8)
  - tile_parallelism: Number of tiles to process concurrently (default: 4)

Usage:
------
  python task4_raster_stats.py --config_path config.json

Or with CLI overrides:
  python task4_raster_stats.py --config_path config.json --chunk_size 5000

Output:
-------
  - Delta table: {catalog}.{schema}.grid_counts
    Columns: grid_id, built, count, urban, lat, lon, tile_id

OPTIMIZATIONS:
--------------
- Tile-level parallelism: Processes multiple tiles concurrently
- Chunk-level parallelism: 8-worker threading within each tile
- Local staging: Copies tiles to local disk for faster reads
- Windowed reads: Memory-efficient raster access
- Boundary masking: Filters points outside country boundaries
- Expected speedup: 2.5-3x on typical workloads

Building Classification:
------------------------
  built_c raster values mapped to categories:
    11-15: Residential (RES)
    21-25: Commercial/Industrial (COM)

SMOD Urban Classification:
---------------------------
  - 0: Rural
  - 1: Urban
  - 2: Suburban

Notes:
------
  - Skips grid points outside country boundaries
  - Uses Mollweide projection (ESRI:54009) for spatial operations
  - Automatically handles missing SMOD data if use_smod=False
"""

# os.environ["CONFIG_PATH"] = "./config.json"

import sys
import os
import time
import json
import shutil
import traceback
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from shapely.ops import unary_union

from pyspark.sql import SparkSession

from utils_geospatial import read_vector_file
# os.environ["CONFIG_PATH"] = "./config.json"
# -----------------------------------------------------------------------------
# CLI -> env wrapper (keeps backward compatibility with existing invocation)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Config loader and defaults
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "grid_source": None,
    "built_root": None,
    "smod_root": None,
    "output_dir": None,
    "counts_delta_table": None,
    "use_smod": True,
    "include_nodata": True,
    "add_percentages": False,
    "use_boundary_mask": True,
    "admin_path": None,
    "admin_field": "ISO3",
    "admin_value": "IND",
    "tile_footprint_path": None,
    "tile_id_field": "tile_id",
    "target_crs": "ESRI:54009",
    "chunk_size": 5000,
    "max_workers": 8,
    "tile_parallelism": 4,  # NEW: number of tiles to process concurrently
    "stage_to_local": True,
    "local_dir": "/local_disk0/raster_cache",
    "save_per_tile": False,
    "write_mode": "overwrite",
    "test_tile": None
}

def _read_json_path(path: str) -> Dict[str, Any]:
    if path.startswith("dbfs:"):
        local_path = path.replace("dbfs:", "/dbfs", 1)
    else:
        local_path = path
    with open(local_path, "r") as f:
        return json.load(f)

def load_and_apply_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG", "") or os.environ.get("CONFIG_JSON", "")
    if cfg_path:
        try:
            loaded = _read_json_path(cfg_path)
            if not isinstance(loaded, dict):
                raise ValueError("config file must be a JSON object")
            cfg.update(loaded)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{cfg_path}': {e}")

    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            val = os.environ[env_key]
            if key in ("use_smod","include_nodata","add_percentages","use_boundary_mask","stage_to_local","save_per_tile"):
                cfg[key] = str(val).lower() in ("true","1","t","yes")
            elif key in ("chunk_size","max_workers","tile_parallelism"):
                try:
                    cfg[key] = int(val)
                except Exception:
                    pass
            else:
                cfg[key] = val

    cfg["include_nodata"] = True
    cfg["add_percentages"] = False
    cfg["use_boundary_mask"] = True
    cfg["use_smod"] = True

    required = ["grid_source","built_root","smod_root","output_dir","counts_delta_table"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {missing}. Provide them in config file or via env overrides.")

    return cfg

def print_minimal_config(cfg: Dict[str, Any]) -> None:
    print("JOB PARAMETERS (effective):")
    print(" GRID_SOURCE         =", cfg.get("grid_source"))
    print(" BUILT_ROOT          =", cfg.get("built_root"))
    print(" SMOD_ROOT           =", cfg.get("smod_root"))
    print(" OUTPUT_DIR          =", cfg.get("output_dir"))
    print(" COUNTS_DELTA_TABLE  =", cfg.get("counts_delta_table"))
    print(" CHUNK_SIZE          =", cfg.get("chunk_size"))
    print(" MAX_WORKERS         =", cfg.get("max_workers"))
    print(" TILE_PARALLELISM    =", cfg.get("tile_parallelism"), "⭐ NEW")
    if cfg.get("test_tile"):
        print(" TEST_TILE           =", cfg.get("test_tile"))

# -----------------------------------------------------------------------------
# Constants and helpers
# -----------------------------------------------------------------------------
BUILT_CLASSES = np.array([11,12,13,14,15,21,22,23,24,25], dtype=np.uint8)
SMOD_CLASSES  = np.array([0,10,11,12,13,21,22,23,30], dtype=np.uint8)
SMOD_RECLASS_MAP = {30:2, 23:2, 22:1, 21:1, 13:0, 12:0, 11:0, 10:0, 0:0}
NODATA = 255

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
                print(f"  Using cached staged file: {dst}")
            return dst
        shutil.copyfile(src_path, dst)
        if verbose:
            print(f"  Staged raster to local: {dst} ({os.path.getsize(src_path)/1e6:.1f} MB)")
        return dst
    except Exception as e:
        if verbose:
            print(f"  WARN: staging failed ({e}), falling back to original path")
        return src_path

# -----------------------------------------------------------------------------
# Chunk processor (reads windows)
# -----------------------------------------------------------------------------
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
                    print(f"    ERROR reading window for grid {row.get('grid_id')} : {e}")
                continue

            if use_boundary_mask and tile_clip is not None and int(row.get('is_boundary', 0)) == 1:
                _mask_boundary_window(arr, w, src.transform, tile_clip)

            counts_vec, valid_count, nodata_count = _count_window(arr, class_vec)

            meta: Dict[str, Any] = {
                "grid_id": row.grid_id,
                "tile_id": row.tile_id,
                "is_boundary": int(row.is_boundary) if 'is_boundary' in row else 0,
            }
            for col in pass_through_columns:
                if col in row:
                    meta[col] = row[col]

            if dataset == "built_c":
                for ccode, ccnt in zip(class_vec, counts_vec):
                    rec: Dict[str, Any] = dict(meta)
                    rec["built"] = int(ccode)
                    rec["count"] = int(ccnt)
                    rec["order_id"] = 1
                    rec["built_total_valid_pixels"] = int(valid_count)
                    if include_nodata:
                        rec["nodata_pixels"] = int(nodata_count)
                    if add_percentages and valid_count > 0:
                        rec["pct"] = float(ccnt) / float(valid_count)
                    out.append(rec)
            else:
                r0 = r1 = r2 = 0
                for ccode, ccnt in zip(class_vec, counts_vec):
                    grp = SMOD_RECLASS_MAP.get(int(ccode), None)
                    if grp == 0: r0 += int(ccnt)
                    elif grp == 1: r1 += int(ccnt)
                    elif grp == 2: r2 += int(ccnt)
                rec: Dict[str, Any] = dict(meta)
                rec["ruban_reclass_0"] = int(r0)
                rec["ruban_reclass_1"] = int(r1)
                rec["ruban_reclass_2"] = int(r2)
                total_ruban = int(r0 + r1 + r2)
                rec["smod_total_valid_pixels"] = int(total_ruban if total_ruban >= 0 else 0)
                if total_ruban == 0:
                    rec["ruban"] = np.nan
                    rec["urban"] = np.nan
                else:
                    dominant = int(np.argmax([r0, r1, r2]))
                    rec["ruban"] = int(dominant)
                    rec["urban"] = int(dominant)
                if include_nodata:
                    rec["smod_nodata_pixels"] = int(nodata_count)
                if add_percentages and total_ruban > 0:
                    rec["ruban_pct_0"] = float(r0) / float(total_ruban)
                    rec["ruban_pct_1"] = float(r1) / float(total_ruban)
                    rec["ruban_pct_2"] = float(r2) / float(total_ruban)
                out.append(rec)
    return out

# -----------------------------------------------------------------------------
# Per-tile processing
# -----------------------------------------------------------------------------
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

    try:
        sdf_tile = spark.read.table(grid_table).where(f"tile_id = '{tile_id}'")
        tile_rows = sdf_tile.toPandas()
    except Exception as e:
        if verbose:
            print(f"    ERROR reading grid rows for tile {tile_id}: {e}")
        return None

    if tile_rows.empty:
        if verbose:
            print(f"    No grid rows for tile {tile_id}")
        return None

    raster_root = built_root if dataset == "built_c" else smod_root
    raster_path = find_raster_for_tile(tile_id, raster_root)
    if raster_path is None:
        if verbose:
            print(f"    [{dataset}] No raster found for tile {tile_id} under {raster_root}")
        return None

    raster_path = stage_to_local_if_needed(raster_path, local_dir, stage_to_local, verbose)
    class_vec = BUILT_CLASSES if dataset == "built_c" else SMOD_CLASSES

    mask_this = use_boundary_mask
    tile_clip = tile_clip_map.get(tile_id) if mask_this else None

    chunks = [tile_rows.iloc[i:i+chunk_size] for i in range(0, len(tile_rows), chunk_size)]
    if verbose:
        print(f"    [{dataset}] windows={len(tile_rows)} chunks={len(chunks)} mask={mask_this}")

    results: List[Dict[str, Any]] = []

    def worker(ch_df):
        try:
            return _process_chunk(raster_path, ch_df, dataset, class_vec, use_boundary_mask, tile_clip, pass_through_columns, include_nodata, add_percentages, False)
        except Exception as e:
            if verbose:
                print(f"      Worker error: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, ch) for ch in chunks]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.extend(res)

    if not results:
        return None
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Prepare tile clips
# -----------------------------------------------------------------------------
def prepare_tile_clips(grid_table: str, spark: SparkSession, admin_path: str, admin_field: str, admin_value: str, tile_footprint_path: str, tile_id_field: str, target_crs: str, verbose: bool) -> Dict[str, Any]:
    if not admin_path or not tile_footprint_path:
        raise RuntimeError("admin_path and tile_footprint_path are required for boundary masking")
    admin = read_vector_file(admin_path, verbose=verbose)
    if admin_field not in admin.columns:
        raise RuntimeError(f"Admin field {admin_field} not in admin columns")
    sel = admin[admin_field] == admin_value
    sel = admin[sel].to_crs(target_crs)
    if sel.empty:
        raise RuntimeError(f"No admin rows match {admin_field}={admin_value}")
    union_geom = unary_union(sel.geometry)
    tiles = read_vector_file(tile_footprint_path, verbose=verbose).to_crs(target_crs)
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

def coalesce_columns(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    for col in base_cols:
        x = f"{col}_x"
        y = f"{col}_y"
        if x in df.columns and y in df.columns:
            df[col] = df[x].combine_first(df[y])
            df.drop([x, y], axis=1, inplace=True)
        elif x in df.columns:
            df.rename(columns={x: col}, inplace=True)
        elif y in df.columns:
            df.rename(columns={y: col}, inplace=True)
    return df

# -----------------------------------------------------------------------------
# NEW: Single tile wrapper for parallel execution
# -----------------------------------------------------------------------------
def process_single_tile_wrapper(
    tile_id: str,
    tile_idx: int,
    total_tiles: int,
    spark: SparkSession,
    config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """
    Wrapper function that processes one complete tile (built + smod + merge).
    Designed to be called from ThreadPoolExecutor.
    """
    try:
        print(f"\n[Thread] Starting tile {tile_idx}/{total_tiles}: {tile_id}")
        tile_start = time.time()
        
        GRID_SOURCE = config["GRID_SOURCE"]
        BUILT_ROOT = config["BUILT_ROOT"]
        SMOD_ROOT = config["SMOD_ROOT"]
        OUTPUT_DIR = config["OUTPUT_DIR"]
        USE_SMOD = config["USE_SMOD"]
        INCLUDE_NODATA = config["INCLUDE_NODATA"]
        ADD_PERCENTAGES = config["ADD_PERCENTAGES"]
        USE_BOUNDARY_MASK = config["USE_BOUNDARY_MASK"]
        CHUNK_SIZE = config["CHUNK_SIZE"]
        MAX_WORKERS = config["MAX_WORKERS"]
        STAGE_TO_LOCAL = config["STAGE_TO_LOCAL"]
        LOCAL_DIR = config["LOCAL_DIR"]
        SAVE_PER_TILE = config["SAVE_PER_TILE"]
        tile_clip_map = config["tile_clip_map"]
        pass_through_columns = config["pass_through_columns"]
        add_iso_suffix = config["add_iso_suffix"]
        
        tile_frames: List[pd.DataFrame] = []

        # Process built_c
        built_path = os.path.join(BUILT_ROOT, str(tile_id))
        if os.path.isdir(built_path):
            df_built = process_tile(tile_id, spark, GRID_SOURCE, "built_c", BUILT_ROOT, SMOD_ROOT, 
                                   CHUNK_SIZE, MAX_WORKERS, USE_BOUNDARY_MASK, tile_clip_map, 
                                   pass_through_columns, INCLUDE_NODATA, ADD_PERCENTAGES, 
                                   STAGE_TO_LOCAL, LOCAL_DIR, True)
            if df_built is not None:
                tile_frames.append(df_built)
        else:
            print(f"  [Thread] built_c folder not found for tile {tile_id}")

        # Process smod
        if USE_SMOD:
            smod_path = os.path.join(SMOD_ROOT, str(tile_id))
            if os.path.isdir(smod_path):
                df_smod = process_tile(tile_id, spark, GRID_SOURCE, "smod", BUILT_ROOT, SMOD_ROOT,
                                      CHUNK_SIZE, MAX_WORKERS, USE_BOUNDARY_MASK, tile_clip_map,
                                      pass_through_columns, INCLUDE_NODATA, ADD_PERCENTAGES,
                                      STAGE_TO_LOCAL, LOCAL_DIR, True)
                if df_smod is not None:
                    tile_frames.append(df_smod)
            else:
                print(f"  [Thread] smod folder not found for tile {tile_id}")

        if not tile_frames:
            print(f"  [Thread] No data produced for tile {tile_id}")
            return None

        # Merge built and smod
        built_df = None
        ruban_df = None
        for df in tile_frames:
            if "built" in df.columns:
                built_df = df
            elif "ruban_reclass_0" in df.columns:
                ruban_df = df

        if built_df is not None and ruban_df is not None:
            ruban_cols_keep = ["ruban_reclass_0","ruban_reclass_1","ruban_reclass_2","ruban","urban","smod_total_valid_pixels","smod_nodata_pixels"]
            ruban_cols = [c for c in ruban_cols_keep if c in ruban_df.columns]
            ruban_short = ruban_df[["grid_id","tile_id","is_boundary"] + ruban_cols]
            ruban_short = ruban_short.sort_values(["grid_id", "tile_id", "is_boundary"]).drop_duplicates(subset=["grid_id", "tile_id", "is_boundary"], keep="first")
            merged = built_df.merge(ruban_short, on=["grid_id", "tile_id", "is_boundary"], how="left")
        elif built_df is not None:
            merged = built_df.copy()
            for c in ["ruban_reclass_0","ruban_reclass_1","ruban_reclass_2","ruban","urban","smod_total_valid_pixels","smod_nodata_pixels"]:
                merged[c] = np.nan
        else:
            print(f"  [Thread] No built data for tile {tile_id}, skipping")
            return None

        merged = coalesce_columns(merged, pass_through_columns)

        final_cols = ["grid_id","tile_id","lat","lon","built","count","built_total_valid_pixels","smod_total_valid_pixels","urban"]
        for c in final_cols:
            if c not in merged.columns:
                merged[c] = np.nan
        merged = merged[final_cols]

        if SAVE_PER_TILE:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_tile = add_iso_suffix(os.path.join(OUTPUT_DIR, f"class_counts_long_{tile_id}_{ts}.csv"))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            merged.to_csv(out_tile, index=False)
            print(f"  [Thread] Saved per-tile CSV: {out_tile}")

        elapsed = time.time() - tile_start
        print(f"  [Thread] Completed tile {tile_id} in {elapsed:.1f}s ({len(merged):,} rows)")
        return merged

    except Exception as e:
        print(f"  [Thread] ERROR processing tile {tile_id}: {e}")
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    config = load_and_apply_config()
    print_minimal_config(config)
    ISO3 = config.get("iso3", "IND").strip().upper()

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

    # Extract config values
    GRID_SOURCE = add_iso_suffix(config["grid_source"])
    BUILT_ROOT = config["built_root"]
    SMOD_ROOT = config["smod_root"]
    OUTPUT_DIR = config["output_dir"]
    COUNTS_DELTA_TABLE = add_iso_suffix(config["counts_delta_table"])
    USE_SMOD = bool(config["use_smod"])
    INCLUDE_NODATA = bool(config["include_nodata"])
    ADD_PERCENTAGES = bool(config["add_percentages"])
    USE_BOUNDARY_MASK = bool(config["use_boundary_mask"])
    ADMIN_PATH = config.get("admin_path")
    ADMIN_FIELD = config.get("admin_field", "ISO3")
    ADMIN_VALUE = config.get("admin_value", "IND")
    TILE_FOOTPRINT_PATH = config.get("tile_footprint_path")
    TILE_ID_FIELD = config.get("tile_id_field", "tile_id")
    TARGET_CRS = config.get("target_crs", "ESRI:54009")
    CHUNK_SIZE = int(config.get("chunk_size", 5000))
    MAX_WORKERS = int(config.get("max_workers", 8))
    TILE_PARALLELISM = int(config.get("tile_parallelism", 3))  # NEW
    STAGE_TO_LOCAL = bool(config.get("stage_to_local", True))
    LOCAL_DIR = config.get("local_dir", "/local_disk0/raster_cache")
    SAVE_PER_TILE = bool(config.get("save_per_tile", False))
    WRITE_MODE = config.get("write_mode", "overwrite")
    TEST_TILE = config.get("test_tile") or os.environ.get("TEST_TILE")

    spark = SparkSession.builder.getOrCreate()

    if os.path.sep in GRID_SOURCE:
        raise RuntimeError("GRID_SOURCE appears to be a filesystem path; this job requires a Delta table spec.")

    print(f"Listing distinct tile IDs from table: {GRID_SOURCE}")
    tile_ids = spark.read.table(GRID_SOURCE).select("tile_id").distinct().rdd.map(lambda r: r[0]).collect()
    tile_ids = [str(t) for t in tile_ids if t is not None and str(t).strip() != ""]
    if not tile_ids:
        raise RuntimeError(f"No tile_ids found in table {GRID_SOURCE}")
    print(f"Found {len(tile_ids)} unique tiles")

    if TEST_TILE:
        TEST_TILE = str(TEST_TILE)
        if TEST_TILE not in tile_ids:
            print(f"WARNING: TEST_TILE {TEST_TILE} not found; continuing with all tiles")
        else:
            tile_ids = [TEST_TILE]
            print(f"Running test for single tile: {TEST_TILE}")

    tile_clip_map: Dict[str, Any] = {}
    if USE_BOUNDARY_MASK:
        tile_clip_map = prepare_tile_clips(GRID_SOURCE, spark, ADMIN_PATH, ADMIN_FIELD, ADMIN_VALUE, 
                                           TILE_FOOTPRINT_PATH, TILE_ID_FIELD, TARGET_CRS, verbose=True)

    pass_through_columns = ["centroid_x","centroid_y","lon","lat","i_idx","j_idx"]

    # Package config for workers
    worker_config = {
        "GRID_SOURCE": GRID_SOURCE,
        "BUILT_ROOT": BUILT_ROOT,
        "SMOD_ROOT": SMOD_ROOT,
        "OUTPUT_DIR": OUTPUT_DIR,
        "USE_SMOD": USE_SMOD,
        "INCLUDE_NODATA": INCLUDE_NODATA,
        "ADD_PERCENTAGES": ADD_PERCENTAGES,
        "USE_BOUNDARY_MASK": USE_BOUNDARY_MASK,
        "CHUNK_SIZE": CHUNK_SIZE,
        "MAX_WORKERS": MAX_WORKERS,
        "STAGE_TO_LOCAL": STAGE_TO_LOCAL,
        "LOCAL_DIR": LOCAL_DIR,
        "SAVE_PER_TILE": SAVE_PER_TILE,
        "tile_clip_map": tile_clip_map,
        "pass_through_columns": pass_through_columns,
        "add_iso_suffix": add_iso_suffix
    }

    print(f"\n{'='*60}")
    print(f"STARTING PARALLEL TILE PROCESSING")
    print(f"Total tiles: {len(tile_ids)}")
    print(f"Tile parallelism: {TILE_PARALLELISM} workers")
    print(f"Per-tile chunk workers: {MAX_WORKERS}")
    print(f"{'='*60}\n")

    outputs: List[pd.DataFrame] = []
    
    # NEW: Parallel tile processing
    with ThreadPoolExecutor(max_workers=TILE_PARALLELISM) as executor:
        # Submit all tiles
        future_to_tile = {
            executor.submit(process_single_tile_wrapper, tile, idx, len(tile_ids), spark, worker_config): tile
            for idx, tile in enumerate(sorted(tile_ids), start=1)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_tile):
            tile = future_to_tile[future]
            try:
                result = future.result()
                if result is not None:
                    outputs.append(result)
            except Exception as e:
                print(f"ERROR: Tile {tile} raised exception: {e}")
                traceback.print_exc()

    if not outputs:
        print("No tiles produced data. Exiting.")
        return

    # Concatenate and write
    print(f"\n{'='*60}")
    print("Combining results from all tiles...")
    combined = pd.concat(outputs, ignore_index=True, sort=False)

    ts_all = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_all = add_iso_suffix(os.path.join(OUTPUT_DIR, f"class_counts_all_long_{ts_all}.csv"))
    combined.to_csv(out_all, index=False)
    print(f"Saved combined counts CSV: {out_all} ({len(combined):,} rows)")

    # Write to Delta
    try:
        print(f"Writing to Delta table: {COUNTS_DELTA_TABLE} (mode={WRITE_MODE})")
        sdf = spark.createDataFrame(combined)
        writer = sdf.write.format("delta").mode(WRITE_MODE)
        if WRITE_MODE == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(COUNTS_DELTA_TABLE)
        print("Successfully saved to Delta table.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Warning: Delta write failed: {e}\n{tb}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"JOB COMPLETED in {elapsed:.1f}s")
    print(f"Expected speedup vs sequential: ~{TILE_PARALLELISM * 0.85:.1f}x")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
