"""
Class counting pipeline for GHSL built_c and smod rasters over 5 km grid cells.
Directory structure expected:

data/
  india_5km_grids_complete.csv   (or centroids file)
  tiles/
    built_c/<TILE_ID>/GHS_BUILT_C_MSZ_E2018_..._<TILE_ID>.tif
    smod/<TILE_ID>/GHS_SMOD_E2020_..._<TILE_ID>.tif
  class_counts_output/           (created if missing)

Features:
- Auto-detect tiles from grid CSV (tile_id column).
- Uses np.bincount for fastest categorical counts (confirmed by benchmarks).
- No percentages by default (toggleable).
- Supports per-tile parallel processing (multiprocessing within a tile).
- Produces one wide CSV consolidating all tiles processed.
"""

import os
import time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import psutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration Constants
# -----------------------------

# Classes (uint8 codes) per dataset
BUILT_CLASSES = np.array([0,1,2,3,4,5,11,12,13,14,15,21,22,23,24,25], dtype=np.uint8)
SMOD_CLASSES  = np.array([0,10,11,12,13,21,22,23,30], dtype=np.uint8)
NODATA_VALUE  = 255

# Adjust if your cell size differs
GRID_CELL_SIZE_M = 5000  # 5 km

# -----------------------------
# Utility Functions
# -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def derive_bounds_if_missing(grid_df: pd.DataFrame,
                             cell_size: int = GRID_CELL_SIZE_M) -> pd.DataFrame:
    """Ensure grid_minx/... columns exist (derive from centroid if absent)."""
    required = {'grid_minx','grid_miny','grid_maxx','grid_maxy'}
    if required.issubset(grid_df.columns):
        return grid_df
    half = cell_size / 2
    grid_df = grid_df.copy()
    grid_df['grid_minx'] = grid_df['centroid_x'] - half
    grid_df['grid_miny'] = grid_df['centroid_y'] - half
    grid_df['grid_maxx'] = grid_df['centroid_x'] + half
    grid_df['grid_maxy'] = grid_df['centroid_y'] + half
    return grid_df

def count_classes(flat_uint8: np.ndarray,
                  class_vector: np.ndarray,
                  nodata: int = NODATA_VALUE):
    """
    Fast per-class counts using a single np.bincount pass.
    Returns (counts vector, valid_pixel_count, nodata_pixel_count)
    """
    nodata_mask = (flat_uint8 == nodata)
    if nodata_mask.all():
        return np.zeros(len(class_vector), dtype=np.int32), 0, int(flat_uint8.size)
    valid = flat_uint8[~nodata_mask]
    if valid.size == 0:
        return np.zeros(len(class_vector), dtype=np.int32), 0, int(nodata_mask.sum())
    bc = np.bincount(valid, minlength=256)
    counts = bc[class_vector]
    return counts.astype(np.int32), int(valid.size), int(nodata_mask.sum())

def discover_raster_path(base_dir: str,
                         dataset: str,
                         tile_id: str) -> str:
    """
    Expecting:
        data/tiles/<dataset>/<tile_id>/*.tif
    """
    candidate_dir = Path(base_dir) / dataset / tile_id
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Directory not found for {dataset} tile {tile_id}: {candidate_dir}")
    tifs = list(candidate_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif in {candidate_dir}")
    # If multiple, choose first (or refine logic if needed)
    return str(tifs[0])

# -----------------------------
# Worker Logic
# -----------------------------

def worker_batch(args):
    """
    Process a batch (chunk) of grids for a single dataset & tile.
    """
    (chunk_df, raster_path, dataset, class_vec,
     include_nodata, compute_dominant) = args

    results = []
    try:
        with rasterio.open(raster_path) as src:
            transform = src.transform
            for _, row in chunk_df.iterrows():
                w = from_bounds(row.grid_minx, row.grid_miny,
                                row.grid_maxx, row.grid_maxy,
                                transform)
                data = src.read(1, window=w)
                flat = data.ravel().astype(np.uint8, copy=False)

                counts_vec, valid_count, nodata_count = count_classes(flat, class_vec)

                rec = {
                    'grid_id': row.grid_id,
                    'tile_id': row.tile_id,
                    'centroid_x': row.centroid_x,
                    'centroid_y': row.centroid_y,
                    f'{dataset}_total_valid_pixels': valid_count
                }
                if include_nodata:
                    rec[f'{dataset}_nodata_pixels'] = nodata_count

                # Add class count columns
                for cls_val, cnt in zip(class_vec, counts_vec):
                    rec[f'{dataset}_class_{cls_val}'] = int(cnt)

                if compute_dominant:
                    if valid_count > 0:
                        max_idx = counts_vec.argmax()
                        rec[f'{dataset}_dominant_class'] = int(class_vec[max_idx])
                        rec[f'{dataset}_dominant_count'] = int(counts_vec[max_idx])
                    else:
                        rec[f'{dataset}_dominant_class'] = np.nan
                        rec[f'{dataset}_dominant_count'] = 0

                results.append(rec)
    except Exception as e:
        print(f"[Worker ERROR dataset={dataset}] {e}")
    return results

# -----------------------------
# Main Class
# -----------------------------

class ClassCountPipeline:
    def __init__(self,
                 grid_csv_path: str,
                 tiles_base_dir: str = "data/tiles",
                 output_dir: str = "data/class_counts_output",
                 max_workers: int | None = None):

        if not os.path.exists(grid_csv_path):
            raise FileNotFoundError(f"Grid CSV not found: {grid_csv_path}")

        self.grid_csv_path = grid_csv_path
        self.tiles_base_dir = tiles_base_dir
        self.output_dir = output_dir
        ensure_dir(self.output_dir)

        # Load grids
        self.grid_df = pd.read_csv(self.grid_csv_path)
        required_cols = {'grid_id','tile_id','centroid_x','centroid_y'}
        missing = required_cols - set(self.grid_df.columns)
        if missing:
            raise ValueError(f"Missing required columns in grid CSV: {missing}")

        self.grid_df = derive_bounds_if_missing(self.grid_df)
        self.tiles = sorted(self.grid_df['tile_id'].unique())

        # Worker count
        logical = psutil.cpu_count(logical=True)
        if max_workers:
            self.max_workers = max_workers
        else:
            self.max_workers = min(8, max(1, logical - 1))

        print(f"✅ Loaded {len(self.grid_df):,} grids across {len(self.tiles)} tiles.")
        print(f"🧠 Using {self.max_workers} workers.")

    def process_tile(self,
                     tile_id: str,
                     datasets=('built_c',),
                     include_nodata=True,
                     compute_dominant=False,
                     chunk_size=250):
        """
        Process a single tile's grids for the specified datasets.
        Each dataset is processed in parallel chunks.
        """
        tile_grids = self.grid_df[self.grid_df['tile_id'] == tile_id].copy()
        if tile_grids.empty:
            print(f"⚠️ Tile {tile_id}: no grids")
            return None

        print(f"\n🔷 Tile {tile_id} | {len(tile_grids)} grids | Datasets: {datasets}")
        dataset_frames = []

        for ds in datasets:
            start_ds = time.time()
            if ds == 'built_c':
                class_vec = BUILT_CLASSES
            elif ds == 'smod':
                class_vec = SMOD_CLASSES
            else:
                print(f"⚠️ Unsupported dataset {ds}, skipping.")
                continue

            try:
                raster_path = discover_raster_path(self.tiles_base_dir, ds, tile_id)
            except FileNotFoundError as e:
                print(f"❌ {e}")
                continue

            print(f"   📂 {ds} raster: {raster_path}")

            # Split into chunks
            chunks = [tile_grids.iloc[i:i+chunk_size]
                      for i in range(0, len(tile_grids), chunk_size)]
            print(f"   📦 {ds}: {len(chunks)} chunks (chunk_size={chunk_size})")

            args_list = [
                (chunk, raster_path, ds, class_vec,
                 include_nodata, compute_dominant)
                for chunk in chunks
            ]

            records = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(worker_batch, a): idx
                           for idx, a in enumerate(args_list)}
                done = 0
                for f in as_completed(futures):
                    batch = f.result()
                    if batch:
                        records.extend(batch)
                    done += 1
                    if done % max(1, len(chunks)//5) == 0:
                        print(f"      Progress {done}/{len(chunks)} ({done/len(chunks)*100:.1f}%)")

            elapsed_ds = time.time() - start_ds
            if records:
                df_ds = pd.DataFrame(records)
                dataset_frames.append(df_ds)
                print(f"   ✅ {ds} done in {elapsed_ds:.2f}s "
                      f"({len(records)/elapsed_ds:.1f} grid-recs/sec)")
            else:
                print(f"   ❌ No records for {ds}")

        if not dataset_frames:
            return None

        # Merge frames on key columns
        merged = dataset_frames[0]
        for extra in dataset_frames[1:]:
            merged = merged.merge(
                extra,
                on=['grid_id','tile_id','centroid_x','centroid_y'],
                how='outer'
            )

        return merged

    def process_all_tiles(self,
                          datasets=('built_c',),
                          include_nodata=True,
                          compute_dominant=False,
                          chunk_size=250,
                          save_per_tile=True):
        """
        Process all tiles sequentially (each tile uses parallel workers).
        """
        all_results = []
        overall_start = time.time()
        for i, tile_id in enumerate(self.tiles):
            print(f"\n=== [{i+1}/{len(self.tiles)}] TILE {tile_id} ===")
            tile_df = self.process_tile(
                tile_id,
                datasets=datasets,
                include_nodata=include_nodata,
                compute_dominant=compute_dominant,
                chunk_size=chunk_size
            )
            if tile_df is None:
                print(f"   ⚠️ Tile {tile_id} produced no results.")
                continue
            if save_per_tile:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                out_tile = os.path.join(
                    self.output_dir,
                    f"class_counts_{tile_id}_{ts}.csv"
                )
                tile_df.to_csv(out_tile, index=False)
                print(f"   💾 Saved per-tile: {out_tile}")
            all_results.append(tile_df)

        if not all_results:
            print("❌ No tiles processed successfully.")
            return None

        combined = pd.concat(all_results, ignore_index=True)
        elapsed = time.time() - overall_start
        print(f"\n✅ ALL COMPLETE in {elapsed/60:.2f} min "
              f"({len(combined)/elapsed:.1f} grid rows/sec)")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.output_dir, f"class_counts_all_{ts}.csv")
        combined.to_csv(out_path, index=False)
        print(f"💾 Consolidated output: {out_path}")
        return combined

# -----------------------------
# Example CLI Entry
# -----------------------------

if __name__ == "__main__":
    GRID_CSV = "data/india_5km_grids_complete.csv"  # adjust if using centroids-only
    TILES_BASE = "data/tiles"                       # root per your structure
    OUTPUT_DIR = "data/class_counts_output"

    pipeline = ClassCountPipeline(
        grid_csv_path=GRID_CSV,
        tiles_base_dir=TILES_BASE,
        output_dir=OUTPUT_DIR,
        max_workers=None  # auto
    )

    # 1. Test a single tile first:
    if pipeline.tiles:
        test_tile = pipeline.tiles[0]
        test_df = pipeline.process_tile(
            tile_id=test_tile,
            datasets=('built_c','smod'),     # add 'smod' when ready
            include_nodata=True,
            compute_dominant=True,
            chunk_size=250
        )
        if test_df is not None:
            print("\nSample rows from test tile:")
            print(test_df.head().to_string())

    # 2. Full run (uncomment when satisfied)
    combined_df = pipeline.process_all_tiles(
        datasets=('built_c','smod'),
        include_nodata=True,
        compute_dominant=True,
        chunk_size=250,
        save_per_tile=True
    )
