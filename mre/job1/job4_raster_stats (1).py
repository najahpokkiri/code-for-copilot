


#!/usr/bin/env python3
"""
Job 4 (Databricks-ready): Raster counts extraction — long built output + ruban (renamed smod reclass)

Features in this final script:
- Loads JSON config from CONFIG_PATH (dbfs:/ or local) and allows env/CLI overrides.
- Forces include_nodata=True, add_percentages=False, use_boundary_mask=True, use_smod=True.
- Emits built output in long format (one row per grid × built class) with built_total_valid_pixels and keeps zero-count rows.
- Processes smod to compute ruban_reclass_0/1/2, smod_total_valid_pixels and a dominant ruban value (0/1/2) per grid row.
  - If total smod valid pixels == 0 -> ruban = NaN (unknown).
- Applies the same boundary mask to both built and smod windows (uses tile_clip_map).
- Left-joins ruban aggregation onto built_long rows by keys ["grid_id","tile_id","is_boundary"] and repeats ruban and smod totals on each built row.
- Supports TEST_TILE override for quick single-tile tests.
- Cleans column redundancy and selects a stable final column set before writing to CSV/Delta.
- NOTE: This script collects per-tile results into pandas and concatenates at the end. For very large runs, consider writing per-tile results to a Delta staging table (append) to avoid large driver memory use.

Usage:
 - Upload job4_config.json to DBFS (e.g. dbfs:/configs/job4_config.json).
 - Run as a Databricks job with the single parameter:
     ["--config_path","dbfs:/configs/job4_config.json"]
 - You can override single keys with additional CLI params:
     ["--config_path","dbfs:/configs/job4_config.json","--test_tile","R6_C26"]
"""

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
# os.environ["CONFIG_PATH"] = "./config.json"   # loader maps this to /dbfs/tmp/...
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
    "include_nodata": True,      # forced
    "add_percentages": False,    # forced
    "use_boundary_mask": True,   # forced
    "admin_path": None,
    "admin_field": "ISO3",
    "admin_value": "IND",
    "tile_footprint_path": None,
    "tile_id_field": "tile_id",
    "target_crs": "ESRI:54009",
    "chunk_size": 5000,
    "max_workers": 8,
    "stage_to_local": True,
    "local_dir": "/local_disk0/raster_cache",
    "save_per_tile": False,
    "write_mode": "overwrite",
    "test_tile": None  # optional override for single-tile debug
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

    # Override from env (wrapper maps --key to UPPERCASE env)
    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            val = os.environ[env_key]
            if key in ("use_smod","include_nodata","add_percentages","use_boundary_mask","stage_to_local","save_per_tile"):
                cfg[key] = str(val).lower() in ("true","1","t","yes")
            elif key in ("chunk_size","max_workers"):
                try:
                    cfg[key] = int(val)
                except Exception:
                    pass
            else:
                cfg[key] = val

    # Force booleans per spec (explicit)
    cfg["include_nodata"] = True
    cfg["add_percentages"] = False
    cfg["use_boundary_mask"] = True
    cfg["use_smod"] = True

    # Validate required keys
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
    print(" INCLUDE_NODATA      =", cfg.get("include_nodata"))
    print(" ADD_PERCENTAGES     =", cfg.get("add_percentages"))
    print(" USE_BOUNDARY_MASK   =", cfg.get("use_boundary_mask"))
    print(" CHUNK_SIZE          =", cfg.get("chunk_size"))
    print(" MAX_WORKERS         =", cfg.get("max_workers"))
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

# -----------------------------------------------------------------------------
# Chunk processor (reads windows) - emits long for built_c and aggregated smod rows
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
                    print(f"  ERROR reading window for grid {row.get('grid_id')} : {e}")
                continue

            # Apply boundary mask when appropriate (same mask for both datasets)
            if use_boundary_mask and tile_clip is not None and int(row.get('is_boundary', 0)) == 1:
                _mask_boundary_window(arr, w, src.transform, tile_clip)

            counts_vec, valid_count, nodata_count = _count_window(arr, class_vec)

            # Common metadata
            meta: Dict[str, Any] = {
                "grid_id": row.grid_id,
                "tile_id": row.tile_id,
                "is_boundary": int(row.is_boundary) if 'is_boundary' in row else 0,
            }
            for col in pass_through_columns:
                if col in row:
                    meta[col] = row[col]

            if dataset == "built_c":
                # Emit long rows: one per built class
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
                # smod: aggregate into ruban groups and emit one row per grid
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
                # If no valid smod pixels, mark ruban unknown (NaN). Otherwise pick argmax as dominant.
                if total_ruban == 0:
                    rec["ruban"] = np.nan
                    rec["urban"] = np.nan
                else:
                    dominant = int(np.argmax([r0, r1, r2]))
                    rec["ruban"] = int(dominant)
                    rec["urban"] = 1 if dominant != 0 else 0
                if include_nodata:
                    rec["smod_nodata_pixels"] = int(nodata_count)
                if add_percentages and total_ruban > 0:
                    rec["ruban_pct_0"] = float(r0) / float(total_ruban)
                    rec["ruban_pct_1"] = float(r1) / float(total_ruban)
                    rec["ruban_pct_2"] = float(r2) / float(total_ruban)
                out.append(rec)
    return out

# -----------------------------------------------------------------------------
# Per-tile processing (Spark -> pandas per tile)
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

    # Use same mask logic for both datasets if enabled
    mask_this = use_boundary_mask
    tile_clip = tile_clip_map.get(tile_id) if mask_this else None

    # split into chunks
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
                if done % max(1, max(1, len(chunks)//5)) == 0 or done == len(chunks):
                    print(f"    [{dataset}] chunk progress {done}/{len(chunks)} ({pct:.1f}%)")

    if not results:
        return None
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Prepare tile clips (boundary masking)
# -----------------------------------------------------------------------------
def prepare_tile_clips(grid_table: str, spark: SparkSession, admin_path: str, admin_field: str, admin_value: str, tile_footprint_path: str, tile_id_field: str, target_crs: str, verbose: bool) -> Dict[str, Any]:
    if not admin_path or not tile_footprint_path:
        raise RuntimeError("admin_path and tile_footprint_path are required for boundary masking")
    admin = gpd.read_file(admin_path)
    if admin_field not in admin.columns:
        raise RuntimeError(f"Admin field {admin_field} not in admin columns")
    sel = admin[admin_field] == admin_value
    sel = admin[sel].to_crs(target_crs)
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

# -----------------------------------------------------------------------------
# Helpers to clean merged DataFrame columns (coalesce duplicates and final select)
# -----------------------------------------------------------------------------
def coalesce_columns(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    # Coalesce _x/_y style duplicates into single base column name
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
# Main workflow
# -----------------------------------------------------------------------------
def main():
    start_time = time.time()
    config = load_and_apply_config()
    print_minimal_config(config)
    ISO3 = config.get("iso3", "IND").strip().upper()

    def add_iso_suffix(name, iso3=ISO3):
        """
        Appends _ISO3 to a table or file name if not already present.
        Handles .csv/.parquet endings for files.
        """
        if name is None:
            return name
        if name.upper().endswith(f"_{iso3}"):
            return name
        if name.endswith(".csv"):
            return name[:-4] + f"_{iso3}.csv"
        if name.endswith(".parquet"):
            return name[:-8] + f"_{iso3}.parquet"
        return f"{name}_{iso3}"


    # Set variables from config
    GRID_SOURCE = add_iso_suffix(config["grid_source"])
    BUILT_ROOT = config["built_root"]
    SMOD_ROOT = config["smod_root"]
    OUTPUT_DIR = config["output_dir"]
    # COUNTS_DELTA_TABLE = config["counts_delta_table"]
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
    STAGE_TO_LOCAL = bool(config.get("stage_to_local", True))
    LOCAL_DIR = config.get("local_dir", "/local_disk0/raster_cache")
    SAVE_PER_TILE = bool(config.get("save_per_tile", False))
    WRITE_MODE = config.get("write_mode", "overwrite")
    TEST_TILE = config.get("test_tile") or os.environ.get("TEST_TILE")

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

    # If requested, limit to a single test tile
    if TEST_TILE:
        TEST_TILE = str(TEST_TILE)
        if TEST_TILE not in tile_ids:
            print(f"WARNING: requested TEST_TILE {TEST_TILE} not found in GRID_SOURCE; continuing with all tiles")
        else:
            tile_ids = [TEST_TILE]
            print(f"Running test for single tile: {TEST_TILE}")

    # Prepare boundary clips if required
    tile_clip_map: Dict[str, Any] = {}
    if USE_BOUNDARY_MASK:
        tile_clip_map = prepare_tile_clips(GRID_SOURCE, spark, ADMIN_PATH, ADMIN_FIELD, ADMIN_VALUE, TILE_FOOTPRINT_PATH, TILE_ID_FIELD, TARGET_CRS, verbose=True)

    pass_through_columns = ["centroid_x","centroid_y","lon","lat","i_idx","j_idx"]
    outputs: List[pd.DataFrame] = []

    print(f"Processing {len(tile_ids)} tiles | workers={MAX_WORKERS} chunk_size={CHUNK_SIZE} stage_to_local={STAGE_TO_LOCAL}")

    for idx, tile in enumerate(sorted(tile_ids), start=1):
        print(f"\n=== Tile {idx}/{len(tile_ids)}: {tile} ===")
        tile_frames: List[pd.DataFrame] = []

        # built_c (long)
        built_path = os.path.join(BUILT_ROOT, str(tile))
        if os.path.isdir(built_path):
            df_built = process_tile(tile, spark, GRID_SOURCE, "built_c", BUILT_ROOT, SMOD_ROOT, CHUNK_SIZE, MAX_WORKERS, USE_BOUNDARY_MASK, tile_clip_map, pass_through_columns, INCLUDE_NODATA, ADD_PERCENTAGES, STAGE_TO_LOCAL, LOCAL_DIR, True)
            if df_built is not None:
                tile_frames.append(df_built)
        else:
            print(f"  built_c folder not found for tile {tile} under {BUILT_ROOT}")

        # smod (ruban aggregation)
        if USE_SMOD:
            smod_path = os.path.join(SMOD_ROOT, str(tile))
            if os.path.isdir(smod_path):
                df_smod = process_tile(tile, spark, GRID_SOURCE, "smod", BUILT_ROOT, SMOD_ROOT, CHUNK_SIZE, MAX_WORKERS, USE_BOUNDARY_MASK, tile_clip_map, pass_through_columns, INCLUDE_NODATA, ADD_PERCENTAGES, STAGE_TO_LOCAL, LOCAL_DIR, True)
                if df_smod is not None:
                    tile_frames.append(df_smod)
            else:
                print(f"  smod folder not found for tile {tile} under {SMOD_ROOT}")

        if not tile_frames:
            print(f"  No data produced for tile {tile}")
            continue

        # Identify built_long and ruban frames
        built_df = None
        ruban_df = None
        for df in tile_frames:
            if "built" in df.columns:
                built_df = df
            elif "ruban_reclass_0" in df.columns or "ruban_reclass_1" in df.columns:
                ruban_df = df

        if built_df is not None and ruban_df is not None:
            # Only keep ruban-specific columns to avoid duplicating pass-through metadata
            ruban_cols_keep = ["ruban_reclass_0","ruban_reclass_1","ruban_reclass_2","ruban","urban","smod_total_valid_pixels","smod_nodata_pixels"]
            ruban_cols = [c for c in ruban_cols_keep if c in ruban_df.columns]
            ruban_short = ruban_df[["grid_id","tile_id","is_boundary"] + ruban_cols]
            # deterministic reduction: keep first if duplicates exist
            ruban_short = ruban_short.sort_values(["grid_id", "tile_id", "is_boundary"]).drop_duplicates(subset=["grid_id", "tile_id", "is_boundary"], keep="first")
            merged = built_df.merge(ruban_short, on=["grid_id", "tile_id", "is_boundary"], how="left")
        elif built_df is not None:
            merged = built_df.copy()
            # ensure columns expected from ruban exist for schema stability
            for c in ["ruban_reclass_0","ruban_reclass_1","ruban_reclass_2","ruban","urban","smod_total_valid_pixels","smod_nodata_pixels"]:
                merged[c] = np.nan
        else:
            # No built data; convert ruban rows to long-like rows with built NaNs
            ruban_df = ruban_df.reset_index(drop=True)
            recs = []
            for _, r in ruban_df.iterrows():
                rec = {
                    "grid_id": r.get("grid_id"),
                    "tile_id": r.get("tile_id"),
                    "is_boundary": int(r.get("is_boundary", 0)),
                    "built": np.nan,
                    "count": np.nan,
                    "order_id": 1,
                }
                for col in pass_through_columns:
                    if col in r:
                        rec[col] = r[col]
                rec["ruban_reclass_0"] = r.get("ruban_reclass_0")
                rec["ruban_reclass_1"] = r.get("ruban_reclass_1")
                rec["ruban_reclass_2"] = r.get("ruban_reclass_2")
                rec["ruban"] = r.get("ruban")
                rec["urban"] = r.get("urban")
                rec["smod_total_valid_pixels"] = r.get("smod_total_valid_pixels")
                rec["smod_nodata_pixels"] = r.get("smod_nodata_pixels")
                recs.append(rec)
            merged = pd.DataFrame(recs)

        # Coalesce any accidental _x/_y duplicate columns (defensive)
        merged = coalesce_columns(merged, pass_through_columns)

        # Final desired columns order and ensure they exist
        # final_cols = [
        #     "grid_id","tile_id",
        #     #"is_boundary",
        #     "centroid_x","centroid_y",
        #     #"lon","lat","i_idx","j_idx",
        #     "built","count",
        #     #"order_id",
        #     "built_total_valid_pixels",
        #     #"nodata_pixels",
        #     # "ruban","ruban_reclass_0","ruban_reclass_1","ruban_reclass_2",
        #     "smod_total_valid_pixels",
        #     #"smod_nodata_pixels",
        #     "urban"
        # ]
        pass_through_columns = ["lat","lon","centroid_x","centroid_y","i_idx","j_idx"]

        final_cols = [
            "grid_id","tile_id",
            # "is_boundary",
            "lat","lon",           # <-- output these, not centroid_x/y
            "built","count",
            "built_total_valid_pixels",
            "smod_total_valid_pixels",
            "urban"
        ]
        for c in final_cols:
            if c not in merged.columns:
                merged[c] = np.nan

        merged = merged[final_cols]

        if SAVE_PER_TILE:
            ts = time.strftime("%Y%m%d_%H%M%S")
            # out_tile = os.path.join(OUTPUT_DIR, f"class_counts_long_{tile}_{ts}.csv")
            out_tile = add_iso_suffix(os.path.join(OUTPUT_DIR, f"class_counts_long_{tile}_{ts}.csv"))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            merged.to_csv(out_tile, index=False)
            print(f"  Saved per-tile CSV (long): {out_tile}")

        outputs.append(merged)

    if not outputs:
        print("No tiles produced data. Exiting.")
        return

    # Concatenate and write combined
    combined = pd.concat(outputs, ignore_index=True, sort=False)

    ts_all = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # out_all = os.path.join(OUTPUT_DIR, f"class_counts_all_long_{ts_all}.csv")
    out_all = add_iso_suffix(os.path.join(OUTPUT_DIR, f"class_counts_all_long_{ts_all}.csv"))
    combined.to_csv(out_all, index=False)
    print(f"\nSaved combined counts CSV: {out_all} rows={len(combined):,}")

    # Write to Delta
    try:
        print(f"Attempting to write combined counts to Delta table: {COUNTS_DELTA_TABLE} (mode={WRITE_MODE})")
        sdf = spark.createDataFrame(combined)
        writer = sdf.write.format("delta").mode(WRITE_MODE)
        # If user requested overwrite, allow schema overwrite to avoid schema mismatches (use carefully)
        if WRITE_MODE == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(COUNTS_DELTA_TABLE)
        print("Saved combined counts to Delta.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Warning: failed to write combined counts to Delta: {e}\n{tb}")

    elapsed = time.time() - start_time
    print(f"Job completed in {elapsed:.1f}s")

if __name__ == "__main__":
    main()