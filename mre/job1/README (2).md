# Geospatial Solutions Pipeline (Databricks)
----
**Databricks Workflow Link**: [Access here](https://adb-6685660099993059.19.azuredatabricks.net/jobs/125711920366493/tasks?o=6685660099993059)

**Final Output Table**: [Delta Lake](https://adb-6685660099993059.19.azuredatabricks.net/explore/data/prp_mr_bdap_projects/geospatialsolutions/estimates_combined_ind?o=6685660099993059&activeTab=sample)

---

## 📋 Overview

This pipeline processes Global Human Settlement Layer (GHSL) data to generate building density estimates and Total Sum Insured (TSI) calculations at a 5km grid resolution. The pipeline is implemented as a **Databricks Workflow** consisting of **6 sequential tasks**, processing satellite-derived building data for specified countries.

### Key Features
- **ISO3-aware processing**: Supports multiple countries with isolated outputs
- **2km grid generation** with stable, reproducible cell IDs
- **Building classification** by type (Residential/Commercial/Industrial) and storey levels
- **TSI proportion calculations** for floor space estimation
- **Automated tile downloads** from GHSL public repositories
- **Boundary-aware processing** for accurate edge handling

---
### GHSL Datasets

a bit of on the datasets



## 🏗️ Architecture

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             INPUT DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Proportions CSV    • TSI CSV    • Admin GPKG    • Tile Footprint SHP    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TASK 1: Load Multipliers (task1_proportions_to_delta.py)                    │
│  ├─ Input: Proportions CSV, TSI CSV                                         │
│  └─ Output: proportions_IND, tsi_IND (Delta tables)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TASK 2: Grid Generation (task2_grid_generation.py)                          │
│  ├─ Input: Admin boundaries, Tile footprints, proportions table            │
│  └─ Output: grid_centroids_IND (Delta) + CSV snapshot                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│ TASK 3: Tile Download             │   │ TASK 4: Raster Statistics        │
│ (task3_tile_downloader.py)        │   │ (task4_raster_stats.py)          │
│  ├─ Input: grid_centroids_IND     │   │  ├─ Input: grid_centroids_IND    │
│  ├─ Output: download_status_IND   │   │  │         + downloaded tiles    │
│  └─ Output: GHSL tiles (built/smod)│   │  └─ Output: counts_combined_IND  │
└──────────────────────────────────┘   └──────────────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TASK 5: Post-Processing (task5_post_processing.py)                          │
│  ├─ Input: counts_combined_IND, proportions_IND, tsi_IND                    │
│  └─ Output: estimates_combined_IND (final estimates with TSI)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TASK 6: Create Views (task6_create_views.py)                                │
│  ├─ Input: estimates_combined_IND                                           │
│  └─ Output: tsi_proportions_res_IND, tsi_proportions_com_IND,              │
│             tsi_proportions_ind_IND (SQL Views)                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Table

| Task | Script | Purpose | Key Inputs | Delta Outputs | Key Parameters |
|------|--------|---------|------------|---------------|----------------|
| 1 | `task1_proportions_to_delta.py` | Load multiplier CSVs to Delta | Proportions & TSI CSVs | `proportions_{ISO3}`, `tsi_{ISO3}` | `--proportions_csv_path`, `--tsi_csv_path` |
| 2 | `task2_grid_generation.py` | Generate 5km grid centroids | Admin boundaries, Tile footprints | `grid_centroids_{ISO3}` | `--iso3 IND`, `--cell_size 5000` |
| 3 | `task3_tile_downloader.py` | Download GHSL tiles | Grid centroids | `download_status_{ISO3}` | `--datasets built_c,smod`, `--dry_run` |
| 4 | `task4_raster_stats.py` | Extract building counts | Grid centroids, Raster tiles | `counts_combined_{ISO3}` | `--use_smod True`, `--chunk_size 5000` |
| 5 | `task5_post_processing.py` | Calculate sector estimates | Counts, Proportions, TSI | `estimates_combined_{ISO3}` | `--write_mode overwrite` |
| 6 | `task6_create_views.py` | Create TSI proportion views | Estimates table | Views: `tsi_proportions_{lob}_{ISO3}` | Auto-computed from ISO3 |

### Simplified Task Flow

```
[Proportions CSV] + [TSI CSV]
            │
            ▼
    ╔═══════════════╗
    ║    TASK 1     ║ Load Multipliers → Delta Tables
    ╚═══════════════╝
            │
            ▼
    ╔═══════════════╗
    ║    TASK 2     ║ Generate 5km Grid → grid_centroids_IND
    ╚═══════════════╝
            │
      ┌─────┴─────┐
      ▼           ▼
╔═══════════╗ ╔═══════════╗
║  TASK 3   ║ ║  TASK 4   ║
║ Download  ║ ║  Raster   ║
║   Tiles   ║ ║   Stats   ║
╚═══════════╝ ╚═══════════╝
      │           │
      └─────┬─────┘
            ▼
    ╔═══════════════╗
    ║    TASK 5     ║ Post-Processing → estimates_combined_IND
    ╚═══════════════╝
            │
            ▼
    ╔═══════════════╗
    ║    TASK 6     ║ Create TSI Views → SQL Views for RES/COM/IND
    ╚═══════════════╝
```

---

## 📊 Sample Visualizations


### Class reclassifications

![](/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/data/images/ghsl_data_structure.png)

### Grid Coverage Map (India Example)


### Density wrt different LOB

![](/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/data/images//building_density_IND.png)

### Grids Sample

![](/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/data/images/grid_figure_ind.png)

---



---

## 🐛 Troubleshooting

### Common Issues

1. **Schema Not Found Error**
   ```
   Error: SCHEMA_NOT_FOUND
   Solution: Ensure catalog and schema exist before running
   ```

2. **Tile Download Failures**
   ```
   Error: HTTP 429 (Too Many Requests)
   Solution: Reduce download_concurrency to 2-3
   ```

3. **Memory Issues in Task 4**
   ```
   Error: Java heap space / Driver OOM
   Solution: Reduce chunk_size to 2000, increase driver memory
   ```

4. **Missing Columns in Task 5**
   ```
   Error: Column 'urban' not found
   Solution: Ensure Task 4 completed successfully with use_smod=True
   ```

---

## 🔄 Operational Playbooks

### Adding a New Country

```python
# 1. Update config with new ISO3
ISO3 = "BGD"  # Bangladesh

# 2. Run Tasks 2-6 with new ISO3
# All outputs will be suffixed with _BGD

# 3. Verify outputs
spark.table(f"prp_mr_bdap_projects.geospatialsolutions.estimates_combined_{ISO3}").count()
```

### Reprocessing Failed Tiles

```python
# Check download status
status_df = spark.table("prp_mr_bdap_projects.geospatialsolutions.download_status_IND")
failed_tiles = status_df.filter("status LIKE 'failed%'").select("tile_id").distinct()

# Re-run Task 3 for failed tiles only
# Modify task3 to filter tile_ids list
```

---

## 📊 Output Schema

### Final Table: `estimates_combined_{ISO3}`

| Column | Type | Description |
|--------|------|-------------|
| GRID_ID | string | Unique grid cell identifier |
| order_id | integer | Sequential order ID |
| lat | double | Latitude (WGS84) |
| lon | double | Longitude (WGS84) |
| urban | integer | Urban classification (0/1) |
| storey{N}_RES | double | Residential buildings by storey |
| storey{N}_COM | double | Commercial buildings by storey |
| storey{N}_IND | double | Industrial buildings by storey |
| RES_Buildings_SUM | double | Total residential buildings |
| COM_Buildings_SUM | double | Total commercial buildings |
| IND_Buildings_SUM | double | Total industrial buildings |
| *_TSI_* columns | double | Total Surface Index values |
| *_perc columns | double | Percentage distributions |

![](/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/data/images/output_table_preview.png)

---
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
## 🤝 Contributing

For questions, improvements, or issues:
1. Check existing documentation in `/mnt/skills/public/`
2. Review job logs in Databricks
3. Contact the Geospatial Solutions team

---

## 📄 License & Attribution

This pipeline uses Global Human Settlement Layer (GHSL) data:
- Built-up Classification: GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A
- Settlement Model: GHS_SMOD_E2020_GLOBE_R2023A

**Citation**: Pesaresi, M., Politis, P. (2023). GHS-BUILT-C R2023A - GHS Settlement Characteristics, derived from Sentinel2 composite (2018) and other GHS R2023A data. European Commission, Joint Research Centre (JRC)

---

*Last Updated: November 2024*
*Version: 2.0 (6-Task Pipeline)*