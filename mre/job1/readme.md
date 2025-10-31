# Geospatial Solutions Pipeline (Databricks) 
----
Databricks job link: [Access here](https://adb-6685660099993059.19.azuredatabricks.net/jobs/125711920366493/tasks?o=6685660099993059)

Final output table: [Delta Lake](https://adb-6685660099993059.19.azuredatabricks.net/explore/data/prp_mr_bdap_projects/geospatialsolutions/estimates_combined_ind?o=6685660099993059&activeTab=sample)

---

This document is your end‑to‑end guide for running and maintaining the Geospatial Solutions pipeline as a single Databricks Job with five Tasks. It is written for users new to Databricks and includes:
- What each Task does, how they connect, and why specific methods were chosen
- Exact copy/paste parameters for each Task
- A render‑safe dataflow table and an ASCII diagram that work in Azure DevOps
- Status‑check cells you can run before (and after) coding
- Operational playbooks, performance tips, and troubleshooting
- Detailed considerations and measured impact from prior experiments

Terminology
- Job: the single, overall Databricks Job that orchestrates everything.
- Tasks: the five steps inside the Job (Task 1 .. Task 5).

Environment assumptions (tailored to your setup)
- Catalog: prp_mr_bdap_projects
- Schema: geospatialsolutions (watch the spelling; do not use geospatialolutions)
- Cluster: personal compute, 8 cores, 64 GB RAM
- Storage: mounted volumes for inputs/outputs; raster tiles under a mounted path (e.g., /mnt/data/tiles)
- Tile volume per country: ~12 tiles per dataset (built_c and smod)
- ISO3 flow: you pass ISO3 in Task 2 (Grid). Downstream Tasks use ISO‑suffixed table names that you provide in their parameters.

Contents
- Quick start
- Architecture and dataflow (render‑safe table + ASCII diagram)
- Repository layout
- Prerequisites and permissions
- Setup and first‑run checks (status cells)
- Task‑by‑task guide with copy/paste parameters
- Implementation details, rationale, and optimizations
- Detailed considerations and measured impact (experiments)
- Performance tuning
- Monitoring and troubleshooting
- Operational playbooks (reruns, backfills, country expansion)
- Validation and QA checks
- Naming conventions
- Security and privacy
- Appendix (mapping table creation, status notebooks, sample scripts)
- Open questions (please answer)

--------------------------------------------------------------------------------
Quick start

1) Open Databricks, create one Job with 5 Tasks (Task 1..5) that run in order (Task 2 depends on 1, etc.).
2) Paste the parameters from “Task‑by‑task guide” into each Task’s Parameters field.
3) Run “Setup and first‑run checks” in a notebook to confirm catalog/schema and base tables exist.
4) Run:
   - Task 1 when proportions CSV changes (can be skipped if your base table already exists).
   - Task 2 with your ISO3 (IND/NPL/PAK…).
   - Task 3 dry‑run (validate URLs), then real run to download and extract tiles.
   - Task 4 to compute raster class counts.
   - Task 5 to produce final sector‑level totals and percentages (no per‑storey distribution in current design).

--------------------------------------------------------------------------------
Architecture and dataflow

This Job comprises five Tasks. ISO3 is set in Task 2 and then flows via suffixed outputs. Mermaid is not used so this renders consistently in Azure DevOps and GitHub.

Dataflow table (render‑safe)
- Catalog: prp_mr_bdap_projects
- Schema: geospatialsolutions
- Replace {ISO3} with uppercase ISO code (e.g., IND)

| Step | Task (script)                    | Purpose                                                    | Key Inputs                                                       | Delta Outputs (catalog.schema.table)                                                                   | CSV/Filesystem Outputs                               | Key parameters (examples)                                                                                                                                   |
|------|----------------------------------|------------------------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Task 1 — job1_proportions_to_delta.py | Ingest absolute floor‑count CSV, normalize proportions     | Proportions CSV (header), floor_bins                             | prp_mr_bdap_projects.geospatialsolutions.proportions; prp_mr_bdap_projects.geospatialsolutions.proportions_counts                           | —                                                     | --input_csv, --counts_table, --proportions_table, --floor_bins "1,2,3,4-5,6-9,10-19,20+", --backup true, --write_mode overwrite                            |
| 2    | Task 2 — job2_grid_generation.py | Generate 5km grid centroids clipped to ISO3 boundary       | Admin GPKG, Tile footprint SHP, proportions table (as trigger)   | prp_mr_bdap_projects.geospatialsolutions.grid_centroids_{ISO3}                                                                                | .../india_5km_grid_centroids_{ISO3}.csv              | --iso3 IND, --proportions_path prp_mr_bdap_projects.geospatialsolutions.proportions                                                                          |
| 3    | Task 3 — job3_tile_downloader.py | Download GHSL tiles (built_c, smod), record statuses       | grid_centroids_{ISO3} (Delta)                                    | prp_mr_bdap_projects.geospatialsolutions.download_status_{ISO3}                                                                               | {tiles_root}/{dataset}/{tile_id}/*.tif               | --grid_source prp_mr_bdap_projects.geospatialsolutions.grid_centroids_{ISO3}, --tiles_dest_root /mnt/data/tiles, --datasets built_c,smod, --dry_run True    |
| 4    | Task 4 — job4_raster_stats.py    | Windowed raster reads per grid to compute class counts     | grid_centroids_{ISO3} (Delta), extracted TIFFs                   | prp_mr_bdap_projects.geospatialsolutions.counts_combined_{ISO3}                                                                               | /mnt/data/outputs/{ISO3}/class_counts_all_*.csv      | --grid_source prp_mr_bdap_projects.geospatialsolutions.grid_centroids_{ISO3}, --built_root /mnt/data/tiles/built_c, --smod_root /mnt/data/tiles/smod        |
| 5    | Task 5 — job5_post_processing.py | Aggregate sector‑level totals and percentages (no storey bins) | counts_combined_{ISO3} (Delta), optional mapping table            | prp_mr_bdap_projects.geospatialsolutions.estimates_combined_{ISO3}                                                                            | /dbfs/tmp/job5_debug/{ISO3}/final_counts_csv         | --counts_table prp_mr_bdap_projects.geospatialsolutions.counts_combined_{ISO3}, --out_table prp_mr_bdap_projects.geospatialsolutions.estimates_combined_{ISO3} |

ASCII diagram (wrapped so Azure DevOps keeps monospace)
<pre style="white-space:pre; overflow:auto; font-family:Consolas,Menlo,Monaco,monospace;">
Proportions CSV/Table
       |
       v
+---------------------+
| Task 1: Ingest      |
| proportions CSV     |
| -> proportions      |
| -> proportions_counts
+----------+----------+
           | Delta
           v
prp_mr_bdap_projects.geospatialsolutions.proportions
           |
           v
+---------------------+        +------------------------------+
| Task 2 (ISO3):      |        | Admin GPKG + Tile footprint  |
| Grid generation     | <------+ (mounted volumes)            |
| -> grid_centroids   |        +------------------------------+
|    {ISO3} (Delta)   |
| -> CSV snapshot     |
+----------+----------+
           | Delta
           v
prp_mr_bdap_projects.geospatialsolutions.grid_centroids_{ISO3}
           |
           +------------------------------------+
           |                                    |
           v                                    v
+----------------------+             +----------------------+
| Task 3 (ISO3):       |             | Task 4 (ISO3):       |
| Tile downloader      |             | Raster stats         |
| -> download_status   |             | -> counts_combined   |
|    {ISO3} (Delta)    |             |    {ISO3} (Delta)    |
| -> TIFFs on volume   |             | -> CSV snapshot      |
+----------+-----------+             +----------+-----------+
           | Delta                                 |
           v                                       | Delta
prp_mr_bdap_projects.geospatialsolutions.          v
download_status_{ISO3}                 prp_mr_bdap_projects.geospatialsolutions.
                                       counts_combined_{ISO3}
                                                    |
                                                    v
                                         +----------------------+
                                         | Task 5 (ISO3):       |
                                         | Post-processing      |
                                         | sector totals only   |
                                         | -> estimates_combined|
                                         |    {ISO3} (Delta)    |
                                         | -> CSV snapshot      |
                                         +----------+-----------+
                                                    | Delta
                                                    v
                                         prp_mr_bdap_projects.geospatialsolutions.
                                         estimates_combined_{ISO3}
</pre>

--------------------------------------------------------------------------------
Repository layout

- mre/job1/job1_proportions_to_delta.py — Task 1 (CSV → proportions + counts)
- mre/job1/job2_grid_generation.py — Task 2 (Grid, ISO3)
- mre/job1/job3_tile_downloader.py — Task 3 (GHSL tiles)
- mre/job1/job4_raster_stats.py — Task 4 (Raster windows → counts)
- mre/job1/job5_post_processing.py — Task 5 (sector totals and percentages)

--------------------------------------------------------------------------------
Prerequisites and permissions

- Databricks Runtime: recent version with Python 3 and Spark.
- Unity Catalog access to catalog prp_mr_bdap_projects, schema geospatialsolutions.
- Library dependencies (cluster or task-scoped):
  - numpy, pandas, pyarrow
  - geopandas, shapely, fiona, pyproj
  - rasterio
  - requests
- Volumes/mounts readable for:
  - Admin GPKG and tile footprint SHP
  - Output CSVs
  - Tiles destination (ZIP downloads and extracted TIFFs)
- Note: Tasks avoid CREATE SCHEMA. Writes target the existing geospatialsolutions schema.

--------------------------------------------------------------------------------
Setup and first‑run checks (status cells)

Paste these into a Databricks Python notebook. Share the outputs if anything fails so we can guide next steps.

1) Catalog and schema; base proportions table
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

spark.sql("SHOW CATALOGS").show(truncate=False)
spark.sql("USE CATALOG prp_mr_bdap_projects")
spark.sql("SHOW SCHEMAS").show(truncate=False)

print("Base proportions exists:",
      spark.catalog.tableExists("prp_mr_bdap_projects.geospatialsolutions.proportions"))
```

2) File paths used in Task 2 (grid generation)
```python
import os
admin_path = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg"
tile_fp_path = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp"
print("Admin GPKG exists:", os.path.exists(admin_path), admin_path)
print("Tile footprint SHP exists:", os.path.exists(tile_fp_path), tile_fp_path)
```

3) ISO3‑aware table existence (after Task 2+)
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG prp_mr_bdap_projects")

ISO3 = "IND"
SCH = "geospatialsolutions"
for t in [f"{SCH}.grid_centroids_{ISO3}",
          f"{SCH}.download_status_{ISO3}",
          f"{SCH}.counts_combined_{ISO3}",
          f"{SCH}.estimates_combined_{ISO3}"]:
    fq = f"prp_mr_bdap_projects.{t}"
    e = spark.catalog.tableExists(fq)
    print(f"{fq}: exists={e}")
```

--------------------------------------------------------------------------------
Task‑by‑task guide with copy/paste parameters

Task 1 — Ingest proportions CSV → Delta
Use when you have a new/updated proportions CSV.

Parameters:
["--input_csv","/dbfs/FileStore/proportions_input.csv","--counts_table","prp_mr_bdap_projects.geospatialsolutions.proportions_counts","--proportions_table","prp_mr_bdap_projects.geospatialsolutions.proportions","--floor_bins","1,2,3,4-5,6-9,10-19,20+","--backup","true","--write_mode","overwrite"]

Task 2 — Grid generation (ISO3)
Provide ISO3 here; outputs are suffixed with {ISO3}.

Parameters:
["--proportions_path","prp_mr_bdap_projects.geospatialsolutions.proportions","--iso3","IND"]

Outputs:
- prp_mr_bdap_projects.geospatialsolutions.grid_centroids_IND (Delta)
- .../india_5km_grid_centroids_IND.csv

Task 3 — Tile downloader (built_c, smod)
Dry‑run first to validate URLs. Then set --dry_run False for actual downloads.

Parameters (dry‑run):
["--grid_source","prp_mr_bdap_projects.geospatialsolutions.grid_centroids_IND","--tiles_dest_root","/mnt/data/tiles","--download_status_table","prp_mr_bdap_projects.geospatialsolutions.download_status_IND","--datasets","built_c,smod","--download_concurrency","3","--download_retries","2","--dry_run","True","--write_mode","overwrite"]

Task 4 — Raster stats extraction
Computes class counts per grid cell by reading windows from the extracted rasters.

Parameters:
["--grid_source","prp_mr_bdap_projects.geospatialsolutions.grid_centroids_IND","--built_root","/mnt/data/tiles/built_c","--smod_root","/mnt/data/tiles/smod","--output_dir","/mnt/data/outputs/IND","--counts_delta_table","prp_mr_bdap_projects.geospatialsolutions.counts_combined_IND","--use_smod","True","--include_nodata","False","--add_percentages","True","--use_boundary_mask","False","--chunk_size","5000","--max_workers","8","--stage_to_local","True","--local_dir","/local_disk0/raster_cache","--save_per_tile","False","--write_mode","overwrite"]

Task 5 — Post‑processing (sector totals only; no proportions)
Aggregates built_c_class_* into sectors (RES/COM/IND by default) to produce per‑sector totals and percentages; writes estimates_combined_{ISO3}.

Parameters:
["--counts_table","prp_mr_bdap_projects.geospatialsolutions.counts_combined_IND","--out_table","prp_mr_bdap_projects.geospatialsolutions.estimates_combined_IND","--mapping_table","prp_mr_bdap_projects.geospatialsolutions.built_sector_mapping_IND","--iso3","IND","--write_mode","overwrite","--output_dir","/dbfs/tmp/job5_debug/IND"]

--------------------------------------------------------------------------------
Implementation details, rationale, and optimizations

Why these methods
- Delta tables: ACID, schema evolution, and Databricks SQL compatibility. Using overwriteSchema on development writes prevents schema drift errors and makes re‑runs predictable.
- GeoPandas/Shapely (Task 2): Vector operations are modest in size; GeoPandas is simpler and more debuggable than distributed spatial frameworks for this case. Snapped grid origin ensures stability across runs.
- ThreadPoolExecutor (Tasks 3 and 4): IO‑bound workloads (HTTP downloads, many small raster reads) benefit from lightweight client‑side concurrency without distributed complexity.
- Per‑tile pandas batches (Task 4): Avoid toPandas() on full tables (OOM risk). Reading only the tile’s rows keeps memory/net IO bounded and reproducible.
- Local SSD staging (Task 4): Copying large rasters to /local_disk0 reduces latency for thousands of small window reads.
- Sector totals only in Task 5: Current design avoids per‑storey distributions; it produces sector totals and percentages directly from built_c_class_* with a mapping, simplifying outputs while remaining auditable.

How it optimizes the workflow
- Stable 5km grid generation with snapped bounds yields repeatable cells and IDs, preventing off‑by‑one drifts between runs and simplifying joins.
- Per‑tile windowing reduces Spark shuffles and peak memory footprint on the driver.
- ISO3‑suffixed outputs isolate runs by country while reusing a shared base proportions table for Task 2 triggers.
- Status tables (download_status) and CSV snapshots enable auditing and quick triage.

--------------------------------------------------------------------------------
Detailed considerations and measured impact (experiments)

The table below documents approaches that were tried previously, the new (adopted) approaches, and the measured impact from your runs. These notes remain valuable for future enhancements (e.g., if you decide to re‑introduce grid‑level expectations or normalize input proportions upstream).

| Area | Previous method | New method | Measured impact (your run) | Notes / Next steps |
|------|------------------|------------|-----------------------------|--------------------|
| Grid‑level expectations | Melt → expand all combos → multiple outputs | Group by [grid_id, smod] → stack built → single pivot → matrix multiply | >2 minutes → ~5.2 seconds for 131,298 grids (10 built classes) | Historical. Current Task 5 does not compute expectations. Keep this design if re‑introducing per‑storey outputs. |
| Proportions CSV handling | Use CSV as‑is; rows may != 1.0; missing combos drop | Keep as‑is; zero‑fill missing (smod,built) combos at runtime (no normalize) | Stable compute; missing combos contribute 0 to expected; QA “coverage” vs raw CSV ~99.79% | If expectations return later: normalize rows to 1.0 and add missing combos offline for perfect QA alignment. |
| Raster I/O locality (built) | Many small random window reads from Volumes | Stage rasters to /local_disk0, then window‑read locally | Built (200 reads): 42.960s → 3.691s; avg 214.80 ms → 18.46 ms/read (~11.6x faster); copy ~8.27s per tile | Largest win on built_c; copy paid once per tile; counting logic unchanged. |
| Parallelism (threads) | Default/unbounded threads | max_workers = 8 | Built tile (37,320 windows): 4→96.82s; 8→52.43s; 12→58.04s; 16→57.99s | 8 threads sweet spot; more threads caused I/O contention. |
| Batching (chunk size) | Untuned | chunk_size = 5000 | workers=8: 1,000→93.49s; 5,000→67.43s; 20,000→159.73s | Larger chunks reduce overhead until tasks get too heavy; ~5k worked best. |
| Read strategy (built tiles) | Consider full‑tile read | Keep windowed reads (after staging) | FullLocal built: full read 59.31s + slice 0.30s (59.61s) vs local window microbench 3.691s/200 reads | Full reads slower for large LZW tiles; windowed on local SSD is better. |
| SMOD handling | As‑is | Unchanged; optionally staged for uniformity | Volumes avg 0.91 ms → Local 0.73 ms/read (tiny tile, negligible difference) | SMOD not the bottleneck; staging OK for consistency, but optional. |
| Boundary masking | Off (use_boundary_mask=False) | Keep off; if needed, precompute tile mask once | N/A | If enabling later: rasterize mask once per tile, slice per window (avoid per‑window rasterize). |
| Output shaping | Multiple outputs; attrs dropped during melt | Single grid‑wide CSV; merge grid attrs at end | Cleaner output; preserves centroid_x/centroid_y/lon/lat/tile_id/i_idx/j_idx; simpler downstream joins | Include raw built_c_class_* and total_count for QA if desired. |

Note: Timings are from your recent IND run and will vary with cluster size, storage, and dataset footprint.

--------------------------------------------------------------------------------
Performance tuning

Defaults appropriate for your scale (8 cores, ~12 tiles/dataset)
- Task 3: download_concurrency=3, retries=2. Increase to 4–6 if network allows; observe 429/timeout behavior before raising.
- Task 4: max_workers=8 (matches cores), chunk_size=5000. Reduce chunk_size (e.g., 2000) if driver memory pressure appears. Keep stage_to_local=True for performance.
- Task 5: Sector totals are linear; no special tuning needed beyond general Spark settings.

--------------------------------------------------------------------------------
Monitoring and troubleshooting

Status checks (ISO3 aware)
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG prp_mr_bdap_projects")
ISO3 = "IND"
SCH = "geospatialsolutions"
for t in [f"{SCH}.proportions", f"{SCH}.grid_centroids_{ISO3}", f"{SCH}.download_status_{ISO3}",
          f"{SCH}.counts_combined_{ISO3}", f"{SCH}.estimates_combined_{ISO3}"]:
    fq = f"prp_mr_bdap_projects.{t}"
    e = spark.catalog.tableExists(fq)
    print(f"{fq}: exists={e}")
    if e:
        spark.read.table(fq).limit(3).show(truncate=False)
```

Common issues
- SCHEMA_NOT_FOUND or write errors: Ensure catalog prp_mr_bdap_projects and schema geospatialsolutions. Tasks set catalog context before writes; they do not create schemas.
- Table not found: Check spelling (geospatialsolutions) and ISO3 suffixes. Use the status checks to validate.
- Raster folder missing for a tile: Confirm Task 3 extracted TIFFs under /mnt/data/tiles/{dataset}/{tile_id}.
- Driver memory in Task 4: Lower chunk_size and/or max_workers; ensure /local_disk0 has free space.

--------------------------------------------------------------------------------
Operational playbooks

Rerun a country (ISO3)
- Reuse the same parameters for Tasks 2–5 with your ISO3 suffix. Task 3 can be dry‑run again to validate; it will skip tiles already extracted.

Add a new country
- Task 2 with new ISO3 to generate grid.
- Task 3 dry‑run then run to fetch tiles.
- Task 4 compute counts.
- Task 5 produce sector totals.

Backfill proportions
- Update CSV and run Task 1 (backup=true recommended).
- Current Task 5 does not use proportions; if you decide to re‑introduce per‑storey expectations later, we can provide a compatible Task 5 variant.

--------------------------------------------------------------------------------
Validation and QA checks

Grid integrity
- Grid centroid counts roughly match admin area/25 km² within expectation.
- Spot‑check centroids within admin in GIS (or with GeoPandas sjoin) to validate point‑in‑polygon logic.

Counts sanity
- Task 4: built_c_total_valid_pixels ≈ sum of built_c_class_* per row (minus nodata if excluded).
- Task 5: total_count equals the sum of sector totals across RES/COM/IND per grid.

--------------------------------------------------------------------------------
Naming conventions

- Tables: prp_mr_bdap_projects.geospatialsolutions.{name}_{ISO3}
- Base proportions: prp_mr_bdap_projects.geospatialsolutions.proportions
- CSV snapshots: include ISO3, e.g., india_5km_grid_centroids_{ISO3}.csv, class_counts_all_{ts}.csv

--------------------------------------------------------------------------------
Security and privacy

- GHSL tiles are public HTTP; if your network restricts egress, run Task 3 dry‑run to validate URLs and coordinate with networking.
- Use Databricks Secrets for any credentials if moving tiles to private object storage.
- Ensure least‑privilege on catalogs/schemas; Tasks avoid CREATE SCHEMA by design.

--------------------------------------------------------------------------------
Appendix

Create built→sector mapping table once (optional)
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG prp_mr_bdap_projects")

DEFAULT_MAPPING = {
    11:"RES",12:"RES",13:"RES",14:"RES",15:"RES",16:"RES",
    21:"COM",22:"COM",23:"COM",
    24:"IND",25:"IND",26:"IND"
}
df = spark.createDataFrame([(k,v) for k,v in DEFAULT_MAPPING.items()], ["built","sector"])
df.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable("geospatialsolutions.built_sector_mapping_IND")
print("Created geospatialsolutions.built_sector_mapping_IND")
```

Minimal “are we ready?” check (everything in one cell)
```python
from pyspark.sql import SparkSession
import os
spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG prp_mr_bdap_projects")

ISO3="IND"
admin_path="/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg"
tile_fp="/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp"
print("proportions:", spark.catalog.tableExists("prp_mr_bdap_projects.geospatialsolutions.proportions"))
print("admin:", os.path.exists(admin_path))
print("tile_fp:", os.path.exists(tile_fp))
for t in ["grid_centroids","download_status","counts_combined","estimates_combined"]:
    fq=f"prp_mr_bdap_projects.geospatialsolutions.{t}_{ISO3}"
    print(fq, "exists:", spark.catalog.tableExists(fq))
```

