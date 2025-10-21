
"""
Job 2: Grid generation (parameterized proportions_path)

Behavior:
- Reads a job parameter 'proportions_path' (default =
  "prp_mr_bdap_projects.geospatialsolutions.proportions").
  If the value looks like a table (contains a dot or no leading "/" and no ".csv"),
  we attempt spark.read.table(...) otherwise we treat it as a filesystem path (CSV).
- Loads proportions only to validate the trigger and prints a preview (no join).
- Generates 5km grid centroids for the admin area and saves:
    - CSV at OUTPUT_CENTROIDS_CSV (on Volume)
    - Delta table prp_mr_bdap_projects.geospatialsolutions.grid_centroids
- Safe fallbacks for running outside Databricks (env var or default)
"""
from pyspark.sql import SparkSession
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
import sys
import traceback

# Databricks widgets for Jobs parameters (works in notebooks and Jobs task)
try:
    dbutils.widgets.text("proportions_path", "prp_mr_bdap_projects.geospatialsolutions.proportions")
    proportions_path = dbutils.widgets.get("proportions_path")
except Exception:
    # Not running inside Databricks widgets (e.g., local run). Try env var fallback.
    proportions_path = os.environ.get("PROPORTIONS_PATH", "prp_mr_bdap_projects.geospatialsolutions.proportions")

# Other configuration (you can parameterize these similarly if wanted)
ADMIN_PATH = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg"
TILE_FOOTPRINT_PATH = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp"
GRID_OUTPUT_CSV = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/grid_seed/india_5km_grid_centroids.csv"
DELTA_TABLE = "prp_mr_bdap_projects.geospatialsolutions.grid_centroids"

ADMIN_FIELD = "ISO3"
ADMIN_VALUE = "IND"
TILE_ID_FIELD = "tile_id"
CELL_SIZE = 5000
TARGET_CRS = "ESRI:54009"
EXPORT_CRS = "EPSG:4326"
DRY_RUN = False

spark = SparkSession.builder.getOrCreate()

def load_proportions(path: str):
    """
    Load proportions either from a Delta table (schema.table or catalog.schema.table)
    or from a CSV file path.
    Returns a Spark DataFrame.
    """
    # Heuristic: treat as table if path doesn't look like a filesystem path and has no suffix .csv
    is_table = False
    p = path.strip()
    if os.path.sep not in p and ('.' in p or not p.lower().endswith('.csv')):
        is_table = True
    # Also accept explicit prefix "delta:" or "table:" (optional)
    if p.lower().startswith("table:"):
        is_table = True
        p = p.split(":", 1)[1]
    if p.lower().startswith("delta:"):
        is_table = True
        p = p.split(":", 1)[1]

    try:
        if is_table:
            print(f"Attempting to read proportions from table: {p}")
            df = spark.read.table(p)
        else:
            print(f"Attempting to read proportions from CSV: {p}")
            df = spark.read.csv(p, header=True, inferSchema=True)
        # quick sanity check
        cnt = df.count()
        print(f"Proportions loaded: {cnt} rows. Preview:")
        display_rows = 5 if cnt >= 5 else cnt
        if display_rows > 0:
            try:
                df.show(display_rows, truncate=False)
            except Exception:
                # fallback: collect and print
                print(df.limit(display_rows).toPandas())
        return df
    except Exception as e:
        print("ERROR loading proportions:", e)
        traceback.print_exc()
        raise

def generate_grids(admin_path, tile_fp_path, cell_size, target_crs, export_crs, output_csv):
    # Validate inputs
    if not os.path.exists(admin_path):
        raise FileNotFoundError(f"Admin file not found: {admin_path}")
    if not os.path.exists(tile_fp_path):
        raise FileNotFoundError(f"Tile footprint file not found: {tile_fp_path}")

    print("Loading admin and tiles shapefiles...")
    admin_full = gpd.read_file(admin_path)
    tiles_full = gpd.read_file(tile_fp_path)
    print("Admin columns:", admin_full.columns.tolist())
    print("Tiles columns:", tiles_full.columns.tolist())

    if ADMIN_FIELD not in admin_full.columns:
        raise ValueError(f"ADMIN_FIELD '{ADMIN_FIELD}' not in admin columns.")

    admin_india = admin_full[admin_full[ADMIN_FIELD] == ADMIN_VALUE].copy()
    if admin_india.empty:
        raise ValueError("No admin rows matched India. Check ADMIN_FIELD / ADMIN_VALUE.")

    # Reproject to target CRS
    admin_india_m = admin_india.to_crs(target_crs)
    tiles_m = tiles_full.to_crs(target_crs)

    # Keep only tiles that intersect admin area
    candidate_tiles = tiles_m[tiles_m.geometry.intersects(admin_india_m.unary_union)].copy()
    print("Intersecting tiles:", len(candidate_tiles))
    if candidate_tiles.empty:
        raise RuntimeError("No tiles intersect India in provided footprint.")

    # Compute grid bounding box from tile footprints
    xmin, ymin, xmax, ymax = candidate_tiles.total_bounds

    # Snap origin for stability
    def snap_down(v, step):
        import math
        return math.floor(v / step) * step
    def snap_up(v, step):
        import math
        return math.ceil(v / step) * step

    x0 = snap_down(xmin, cell_size)
    y0 = snap_down(ymin, cell_size)
    x1 = snap_up(xmax, cell_size)
    y1 = snap_up(ymax, cell_size)

    x_centers = np.arange(x0 + cell_size/2, x1, cell_size)
    y_centers = np.arange(y0 + cell_size/2, y1, cell_size)
    print("x centers:", len(x_centers), "y centers:", len(y_centers), "total raw:", len(x_centers) * len(y_centers))

    xx, yy = np.meshgrid(x_centers, y_centers)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    centroids_all = gpd.GeoDataFrame(
        {"centroid_x": flat_x, "centroid_y": flat_y},
        geometry=[Point(xy) for xy in zip(flat_x, flat_y)],
        crs=target_crs
    )
    print("Full mesh cells:", len(centroids_all))

    # Keep centroids within admin
    centroids_in = gpd.sjoin(
        centroids_all,
        admin_india_m[['geometry']],
        how="inner",
        predicate='within'
    ).drop(columns=['index_right'], errors='ignore')
    print("Centroids inside admin:", len(centroids_in))

    # Spatial join with tile footprints to assign tile IDs
    centroids_tile = gpd.sjoin(
        centroids_in,
        candidate_tiles[[TILE_ID_FIELD, 'geometry']],
        how='left',
        predicate='within'
    ).rename(columns={TILE_ID_FIELD: 'tile_id'}).drop(columns=['index_right'], errors='ignore')

    # Resolve duplicates deterministically if any
    if centroids_tile.duplicated(subset=['centroid_x', 'centroid_y']).any():
        centroids_tile = (centroids_tile
                          .sort_values(['centroid_x', 'centroid_y', 'tile_id'])
                          .drop_duplicates(subset=['centroid_x', 'centroid_y'], keep='first'))

    # Stable indices & grid id
    i_index = np.round((centroids_tile['centroid_x'] - (x0 + cell_size/2)) / cell_size).astype(int)
    j_index = np.round((centroids_tile['centroid_y'] - (y0 + cell_size/2)) / cell_size).astype(int)
    centroids_tile['i_idx'] = i_index
    centroids_tile['j_idx'] = j_index
    centroids_tile['grid_id'] = "G_" + centroids_tile['j_idx'].astype(str) + "_" + centroids_tile['i_idx'].astype(str)
    centroids_tile['grid_minx'] = centroids_tile['centroid_x'] - cell_size/2
    centroids_tile['grid_maxx'] = centroids_tile['centroid_x'] + cell_size/2
    centroids_tile['grid_miny'] = centroids_tile['centroid_y'] - cell_size/2
    centroids_tile['grid_maxy'] = centroids_tile['centroid_y'] + cell_size/2

    # Convert to WGS84 lon/lat
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

    # Save CSV
    out_dir = os.path.dirname(output_csv)
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print("Saved CSV:", output_csv)

    # Save to Delta (use Spark)
    try:
        sdf = spark.read.csv(output_csv, header=True, inferSchema=True)
        sdf.write.format("delta").mode("overwrite").saveAsTable(DELTA_TABLE)
        print(f"Saved to Delta table {DELTA_TABLE}")
    except Exception as e:
        print("Warning: could not save to Delta table (permission or other issue):", e)
        traceback.print_exc()

    return len(out_df)

def main():
    print("proportions_path parameter:", proportions_path)
    # load proportions (only as a trigger / validation)
    try:
        props = load_proportions(proportions_path)
    except Exception:
        print("Failed to load proportions — aborting job.")
        sys.exit(10)

    # Now generate grids
    try:
        n = generate_grids(ADMIN_PATH, TILE_FOOTPRINT_PATH, CELL_SIZE, TARGET_CRS, EXPORT_CRS, GRID_OUTPUT_CSV)
        print(f"Grid generation complete: {n} rows saved to {GRID_OUTPUT_CSV}")
    except Exception as e:
        print("Grid generation failed:", e)
        traceback.print_exc()
        sys.exit(20)

if __name__ == "__main__":
    main()

# # Job 2: Load proportions and generate grids from shapefiles

# !pip install geopandas shapely
# # Job 2: Generate grids from shapefiles (no join with proportions)
# # Add to your job as a task. This generates grids on the fly.

# from pyspark.sql import SparkSession
# import geopandas as gpd
# from shapely.geometry import Point
# import numpy as np

# spark = SparkSession.builder.getOrCreate()

# # Inputs
# proportions_path = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/multipliers/input.csv"
# admin_path = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg"
# tile_footprint_path = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp"
# admin_field = "ISO3"
# admin_value = "IND"
# tile_id_field = "tile_id"
# cell_size = 5000
# target_crs = "ESRI:54009"
# export_crs = "EPSG:4326"
# output_csv = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/grid_seed/india_5km_grid_centroids.csv"

# # Step 1: Load proportions (just to trigger the workflow, not used here)
# df_props = spark.read.csv(proportions_path, header=True, inferSchema=True)
# print(f"Loaded proportions to trigger workflow: {df_props.count()} rows")

# # Step 2: Generate grids from shapefiles (your original code)
# admin_full = gpd.read_file(admin_path)
# tiles_full = gpd.read_file(tile_footprint_path)
# admin_india = admin_full[admin_full[admin_field] == admin_value].copy()
# admin_india_m = admin_india.to_crs(target_crs)
# tiles_m = tiles_full.to_crs(target_crs)
# candidate_tiles = tiles_m[tiles_m.geometry.intersects(admin_india_m.unary_union)].copy()

# xmin, ymin, xmax, ymax = candidate_tiles.total_bounds
# x_centers = np.arange(xmin + cell_size/2, xmax, cell_size)
# y_centers = np.arange(ymin + cell_size/2, ymax, cell_size)
# xx, yy = np.meshgrid(x_centers, y_centers)
# flat_x = xx.ravel()
# flat_y = yy.ravel()
# centroids_all = gpd.GeoDataFrame({"centroid_x": flat_x, "centroid_y": flat_y}, geometry=[Point(xy) for xy in zip(flat_x, flat_y)], crs=target_crs)
# centroids_in = gpd.sjoin(centroids_all, admin_india_m[['geometry']], how='inner', predicate='within').drop(columns=['index_right'], errors='ignore')

# centroids_tile = gpd.sjoin(centroids_in, candidate_tiles[[tile_id_field,'geometry']], how='left', predicate='within').rename(columns={tile_id_field:'tile_id'}).drop(columns=['index_right'], errors='ignore')
# i_index = np.round((centroids_tile['centroid_x'] - (xmin + cell_size/2)) / cell_size).astype(int)
# j_index = np.round((centroids_tile['centroid_y'] - (ymin + cell_size/2)) / cell_size).astype(int)
# centroids_tile['i_idx'] = i_index
# centroids_tile['j_idx'] = j_index
# centroids_tile['grid_id'] = "G_" + centroids_tile['j_idx'].astype(str) + "_" + centroids_tile['i_idx'].astype(str)
# centroids_tile['grid_minx'] = centroids_tile['centroid_x'] - cell_size/2
# centroids_tile['grid_maxx'] = centroids_tile['centroid_x'] + cell_size/2
# centroids_tile['grid_miny'] = centroids_tile['centroid_y'] - cell_size/2
# centroids_tile['grid_maxy'] = centroids_tile['centroid_y'] + cell_size/2
# centroids_wgs84 = centroids_tile.to_crs(export_crs)
# centroids_tile['lon'] = centroids_wgs84.geometry.x
# centroids_tile['lat'] = centroids_wgs84.geometry.y

# export_cols = ['grid_id','tile_id','centroid_x','centroid_y','lon','lat','grid_minx','grid_miny','grid_maxx','grid_maxy','i_idx','j_idx']
# centroids_tile[export_cols].to_csv(output_csv, index=False)
# print(f"Saved grids to {output_csv}: {len(centroids_tile)} rows")

# # Step 3: Save grids to Delta
# df_grid = spark.read.csv(output_csv, header=True, inferSchema=True)
# table_name = "prp_mr_bdap_projects.geospatialsolutions.grid_centroids"
# df_grid.write.format("delta").mode("overwrite").saveAsTable(table_name)
# print(f"Saved grids to Delta table {table_name}")

# # Verify
# spark.sql(f"SELECT COUNT(*) FROM {table_name}").show()