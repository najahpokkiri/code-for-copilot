#!/usr/bin/env python3
"""
Task 2 — Generate 2km Grid Centroids for Country

Generates a regular grid of centroids at specified resolution (default 2000m)
covering the intersection of country admin boundaries and GHSL tile footprints.

Configuration:
--------------
Reads from config.json (generated from config.yaml via config_builder.py).
All paths and table names are auto-generated from the YAML configuration.

Required config keys:
  - admin_path: Path to admin boundaries (GeoPackage)
  - tile_footprint_path: Path to GHSL tile footprint (GeoPackage or Shapefile)
  - grid_output_csv: Output path for grid CSV
  - delta_table_base: Base name for Delta table
  - iso3: Country ISO3 code
  - cell_size: Grid cell size in meters (default: 2000)
  - target_crs: Target CRS for processing (default: ESRI:54009 Mollweide)
  - export_crs: Export CRS (default: EPSG:4326 WGS84)

Usage:
------
  python task2_grid_generation.py --config_path config.json

Or with CLI overrides:
  python task2_grid_generation.py --config_path config.json --iso3 USA --cell_size 5000

Output:
-------
  - CSV file with grid centroids: grid_id, centroid_x, centroid_y, tile_id
  - Delta table: {catalog}.{schema}.grid_centroids_{iso3}

Notes:
------
  - Grids are generated only for tiles intersecting country boundaries
  - Cell IDs are stable and reproducible based on Mollweide coordinates
  - Proportions table is loaded to ensure consistency with Task 1
"""

import os
import sys
import json
import math
import traceback
from typing import Dict, Any, Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyspark.sql import SparkSession

from utils_geospatial import normalize_path, read_vector_file

# os.environ["CONFIG_PATH"] = "./config.json"


if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            k = a.lstrip("-").upper()
            v = ""
            if (i + 1) < len(args) and not args[i + 1].startswith("--"):
                v = args[i + 1]
                i += 2
            else:
                i += 1
            if v != "":
                os.environ[k] = v
        else:
            i += 1

DEFAULT_CONFIG: Dict[str, Any] = {
    "proportions_path": None,
    "iso3": "IND",
    "admin_path": None,
    "tile_footprint_path": None,
    "grid_output_csv": None,
    "delta_table_base": None,
    "admin_field": "ISO3",
    "tile_id_field": "tile_id",
    "cell_size": 5000,
    "target_crs": "ESRI:54009",
    "export_crs": "EPSG:4326",
    "dry_run": False
}

def _read_json_path(p: str) -> Dict[str, Any]:
    if p.startswith("dbfs:"):
        local = p.replace("dbfs:", "/dbfs", 1)
    else:
        local = p
    with open(local, "r") as fh:
        return json.load(fh)

def load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG", "") or os.environ.get("CONFIG_JSON", "")
    if cfg_path:
        try:
            loaded = _read_json_path(cfg_path)
            if not isinstance(loaded, dict):
                raise ValueError("config JSON must be an object")
            cfg.update(loaded)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{cfg_path}': {e}")
    for k in list(cfg.keys()):
        env = os.environ.get(k.upper())
        if env is not None and env != "":
            if isinstance(cfg[k], bool):
                cfg[k] = str(env).lower() in ("true","1","t","yes")
            elif isinstance(cfg[k], int):
                try:
                    cfg[k] = int(env)
                except Exception:
                    pass
            else:
                cfg[k] = env
    required = ["admin_path","tile_footprint_path","grid_output_csv","delta_table_base"]
    missing = [r for r in required if not cfg.get(r)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {missing}")
    return cfg

def is_table_like(path: str) -> bool:
    p = path.strip()
    if p.lower().startswith("table:") or p.lower().startswith("delta:"):
        return True
    if os.path.sep not in p and ('.' in p or not p.lower().endswith('.csv')):
        return True
    return False

# Fix: Always add ISO3 suffix to table names before reading as table
def add_iso_suffix(name, iso3):
    if name is None:
        return name
    if name.upper().endswith(f"_{iso3}"):
        return name
    if name.endswith(".csv"):
        return name[:-4] + f"_{iso3}.csv"
    if name.endswith(".parquet"):
        return name[:-8] + f"_{iso3}.parquet"
    return f"{name}_{iso3}"

def load_proportions(spark, ppath: str, iso3: str):
    p = ppath.strip()
    # handle prefixes
    if p.lower().startswith("table:"):
        tbl = p.split(":",1)[1]
        # Config builder already includes ISO3 in table name
        print(f"Reading proportions from table: {tbl}")
        return spark.read.table(tbl)
    if is_table_like(p):
        # Config builder already includes ISO3 in table name
        print(f"Reading proportions from table: {p}")
        return spark.read.table(p)
    resolved = normalize_path(p)
    print(f"Reading proportions from CSV: {resolved}")
    return spark.read.option("header", True).option("inferSchema", True).csv(resolved)

def generate_grids(admin_path: str,
                   tile_fp_path: str,
                   cell_size: int,
                   target_crs: str,
                   export_crs: str,
                   output_csv: str,
                   iso3: str,
                   delta_table: str) -> int:
    admin_path_r = normalize_path(admin_path)
    tile_fp_path_r = normalize_path(tile_fp_path)
    output_csv_r = normalize_path(output_csv)

    if not os.path.exists(admin_path_r):
        raise FileNotFoundError(f"Admin file not found: {admin_path_r}")
    if not os.path.exists(tile_fp_path_r):
        raise FileNotFoundError(f"Tile footprint file not found: {tile_fp_path_r}")

    print("Loading admin and tiles...")
    admin_full = read_vector_file(admin_path_r)
    tiles_full = read_vector_file(tile_fp_path_r)

    admin_field = cfg.get("admin_field", "ISO3")

    if admin_field not in admin_full.columns:
        raise ValueError(f"ADMIN_FIELD '{admin_field}' not present in admin file columns {admin_full.columns.tolist()}")

    admin_iso = admin_full[admin_full[admin_field] == iso3].copy()
    if admin_iso.empty:
        raise ValueError(f"No admin rows matched ISO3={iso3}")

    admin_iso_m = admin_iso.to_crs(target_crs)
    tiles_m = tiles_full.to_crs(target_crs)

    candidate_tiles = tiles_m[tiles_m.geometry.intersects(admin_iso_m.unary_union)].copy()
    if candidate_tiles.empty:
        raise RuntimeError(f"No tiles intersect {iso3}")

    xmin, ymin, xmax, ymax = candidate_tiles.total_bounds

    def snap_down(v, step):
        return math.floor(v / step) * step
    def snap_up(v, step):
        return math.ceil(v / step) * step

    x0 = snap_down(xmin, cell_size)
    y0 = snap_down(ymin, cell_size)
    x1 = snap_up(xmax, cell_size)
    y1 = snap_up(ymax, cell_size)

    x_centers = np.arange(x0 + cell_size/2, x1, cell_size)
    y_centers = np.arange(y0 + cell_size/2, y1, cell_size)

    xx, yy = np.meshgrid(x_centers, y_centers)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    centroids_all = gpd.GeoDataFrame(
        {"centroid_x": flat_x, "centroid_y": flat_y},
        geometry=[Point(xy) for xy in zip(flat_x, flat_y)],
        crs=target_crs
    )

    centroids_in = gpd.sjoin(
        centroids_all,
        admin_iso_m[['geometry']],
        how="inner",
        predicate='within'
    ).drop(columns=['index_right'], errors='ignore')

    centroids_tile = gpd.sjoin(
        centroids_in,
        candidate_tiles[[cfg.get("tile_id_field","tile_id"), 'geometry']],
        how='left',
        predicate='within'
    ).rename(columns={cfg.get("tile_id_field","tile_id"): 'tile_id'}).drop(columns=['index_right'], errors='ignore')

    if centroids_tile.duplicated(subset=['centroid_x','centroid_y']).any():
        centroids_tile = (centroids_tile
                          .sort_values(['centroid_x','centroid_y','tile_id'])
                          .drop_duplicates(subset=['centroid_x','centroid_y'], keep='first'))

    i_index = np.round((centroids_tile['centroid_x'] - (x0 + cell_size/2)) / cell_size).astype(int)
    j_index = np.round((centroids_tile['centroid_y'] - (y0 + cell_size/2)) / cell_size).astype(int)
    centroids_tile['i_idx'] = i_index
    centroids_tile['j_idx'] = j_index
    centroids_tile['grid_id'] = "G_" + centroids_tile['j_idx'].astype(str) + "_" + centroids_tile['i_idx'].astype(str)
    centroids_tile['grid_minx'] = centroids_tile['centroid_x'] - cell_size/2
    centroids_tile['grid_maxx'] = centroids_tile['centroid_x'] + cell_size/2
    centroids_tile['grid_miny'] = centroids_tile['centroid_y'] - cell_size/2
    centroids_tile['grid_maxy'] = centroids_tile['centroid_y'] + cell_size/2

    centroids_wgs84 = centroids_tile.to_crs(export_crs)
    centroids_tile['lon'] = centroids_wgs84.geometry.x
    centroids_tile['lat'] = centroids_wgs84.geometry.y

    EXPORT_COLS = [
        'grid_id', 'tile_id',
        'centroid_x', 'centroid_y', 'lon', 'lat',
        'grid_minx', 'grid_miny', 'grid_maxx', 'grid_maxy',
        'i_idx', 'j_idx'
    ]
    out_df = centroids_tile[EXPORT_COLS].copy()

    out_dir = os.path.dirname(output_csv_r)
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_csv_r, index=False)
    print("Saved CSV:", output_csv_r)

    try:
        sdf = spark.read.csv(output_csv_r, header=True, inferSchema=True)
        sdf.write.format("delta").mode("overwrite").saveAsTable(delta_table)
        print(f"Saved to Delta table {delta_table}")
    except Exception as e:
        print("Warning: could not save to Delta table:", e)
        traceback.print_exc()
    return len(out_df)

if __name__ == "__main__":
    cfg = load_config()
    print("Job2 config (effective):")
    ISO3 = cfg.get("iso3", "IND").strip().upper()
    for k,v in cfg.items():
        print(f" {k:20} = {v}")

    proportions_path = cfg.get("proportions_path")
    ADMIN_PATH = cfg.get("admin_path")
    TILE_FOOTPRINT_PATH = cfg.get("tile_footprint_path")
    GRID_OUTPUT_CSV = add_iso_suffix(cfg.get("grid_output_csv"), ISO3)
    # Config builder already includes ISO3 in table name
    DELTA_TABLE = cfg.get("delta_table_base")
    CELL_SIZE = int(cfg.get("cell_size", 5000))
    TARGET_CRS = cfg.get("target_crs", "ESRI:54009")
    EXPORT_CRS = cfg.get("export_crs", "EPSG:4326")
    DRY_RUN = bool(cfg.get("dry_run", False))

    spark = SparkSession.builder.getOrCreate()

    print("proportions_path parameter:", proportions_path)
    print("ISO3 parameter:", ISO3)

    try:
        props_df = load_proportions(spark, proportions_path, ISO3)
        cnt = props_df.count()
        print(f"Proportions loaded: {cnt} rows (used only as trigger/validation).")
    except Exception as exc:
        print("Failed to load proportions:", exc)
        sys.exit(10)

    if DRY_RUN:
        print("DRY_RUN enabled: skipping generation and writes.")
        sys.exit(0)

    try:
        n = generate_grids(ADMIN_PATH, TILE_FOOTPRINT_PATH, CELL_SIZE, TARGET_CRS, EXPORT_CRS, GRID_OUTPUT_CSV, ISO3, DELTA_TABLE)
        print(f"Grid generation complete: {n} rows saved to {GRID_OUTPUT_CSV} and table {DELTA_TABLE}")
    except Exception as e:
        print("Grid generation failed:", e)
        traceback.print_exc()
        sys.exit(20)