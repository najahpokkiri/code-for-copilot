# Building Enrichment Pipeline - Quick Start Instructions

## Overview
This pipeline processes geospatial building data to generate enriched building inventories with TSI (Total Sum Insured) calculations and export capabilities. Everything runs via a single Jupyter notebook - **no CLI or config files required**.

## Prerequisites

1. **Databricks Workspace** with access to:
   - A cluster with Databricks Runtime 13.3+ LTS
   - Unity Catalog enabled
   - Volumes for data storage

2. **Input Data Required**:
   - Proportions CSV (building type distribution by storey)
   - TSI CSV (Total Sum Insured multipliers)
   - World/Admin boundaries GeoPackage (optional)
   - Tile footprint data (included in repo as `ghsl2_0_mwd_l1_tile_schema_land.gpkg`)

3. **Permissions**:
   - Read/write access to specified catalog and schema
   - Read/write access to specified volumes
   - Ability to create and run jobs on your cluster

## Step-by-Step Instructions

### Step 1: Upload Files to Databricks Workspace

1. Upload the entire `mre/job1/` folder to your Databricks workspace
2. Recommended location: `/Workspace/Users/<your-email>/code-for-copilot/mre/job1/`

### Step 2: Open the Notebook

Open `create_and_run_job.ipynb` in Databricks and attach it to a cluster (Runtime 13.3+ LTS recommended)

### Step 3: Edit Configuration IN the Notebook

In **Cell 2** of the notebook, edit these variables:

```python
# Country code
ISO3 = "USA"  # Change to your country's ISO3 code

# Databricks settings
CATALOG = "prp_mr_bdap_projects"
SCHEMA = "geospatialsolutions"
VOLUME_BASE = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data"

# Input file paths (full paths)
PROPORTIONS_CSV = "/Workspace/Users/npokkiri@munichre.com/data/proportions.csv"
TSI_CSV = "/Volumes/catalog/schema/data/tsi.csv"
ADMIN_BOUNDARIES = "/Volumes/catalog/schema/data/admin/boundaries.gpkg"

# Workspace path (where these scripts are located)
WORKSPACE_BASE = "/Workspace/Users/<your-email>/code-for-copilot/mre/job1"

# Optional: Email for notifications
EMAIL = "your-email@company.com"

# Optional: Cluster ID (leave empty to auto-detect)
CLUSTER_ID = ""  # Will auto-detect current cluster if empty
```

That's it! **No separate config file needed**.

### Step 4: Run All Cells

Run all cells in the notebook. It will automatically:
- ✅ Auto-install required packages (databricks-sdk, pyyaml)
- ✅ Auto-detect cluster ID
- ✅ Generate minimal config
- ✅ Create Databricks job with 8 tasks:
  - **Task 0**: Setup (creates folders, copies files, generates full config)
  - **Task 1-7**: Pipeline execution
- ✅ Run the job
- ✅ Monitor progress in real-time
- ✅ Verify outputs

### Step 5: What Happens in the Job

**Task 0 (Setup)** automatically:
1. Creates `{ISO3}/input/`, `{ISO3}/output/`, `{ISO3}/logs/` folders
2. Copies tiles to `{ISO3}/input/tiles/`
3. Copies your CSVs to `{ISO3}/input/`
4. Uses `config_builder.py` to generate full `config.json` with ISO3 suffixes
5. Saves config to `{ISO3}/config.json`

**Task 1-7** (Pipeline):
- Read `{ISO3}/config.json`
- Execute the complete pipeline
- All tables created with `_{ISO3}` suffix

### Step 6: Monitor Progress

The notebook displays real-time job progress:

```
⏳ Monitoring job progress...

[0s] Job status: PENDING
[30s] Job status: RUNNING
[35s] ⏳ task0_setup: RUNNING
[65s] ✅ task0_setup: TERMINATED
[70s] ⏳ task1_proportions_to_delta: RUNNING
[120s] ✅ task1_proportions_to_delta: TERMINATED
[125s] ⏳ task2_grid_generation: RUNNING
...
✅ Job completed successfully!
   Duration: 45m 23s
```

### Step 7: Access Your Results

After successful completion:

**Delta Tables** (with ISO3 suffix):
- `{catalog}.{schema}.building_enrichment_output_{ISO3}` - Main output
- `{catalog}.{schema}.building_enrichment_proportions_input_{ISO3}` - Input proportions
- `{catalog}.{schema}.building_enrichment_tsi_input_{ISO3}` - TSI multipliers
- `{catalog}.{schema}.grid_centroids_{ISO3}` - Grid centroids
- `{catalog}.{schema}.grid_counts_{ISO3}` - Building counts

**Export Files** (in Volumes):
- `{volume_base}/{ISO3}/output/exports/FULL_{ISO3}/`
  - `building_enrichment_output_{ISO3}_FULL.csv`
  - `building_enrichment_tsi_proportions_{ISO3}_RES_FULL.csv`
  - `building_enrichment_tsi_proportions_{ISO3}_COM_FULL.csv`
  - `building_enrichment_tsi_proportions_{ISO3}_IND_FULL.csv`

**Excel Summary**:
- `{volume_base}/{ISO3}/output/exports/building_summary_country_layout_{ISO3}.xlsx`

**Config File** (auto-generated):
- `{volume_base}/{ISO3}/config.json`

## Folder Structure Created (by Task 0)

```
{ISO3}/                          # e.g., USA/
├── input/
│   ├── tiles/
│   │   ├── ghsl2_0_mwd_l1_tile_schema_land.gpkg
│   │   ├── built_c/  (created during download)
│   │   └── smod/     (created during download)
│   ├── tsi.csv
│   ├── proportions.csv
│   └── admin_boundaries.gpkg (optional)
├── output/
│   └── exports/
│       └── FULL_{ISO3}/
│           ├── building_enrichment_output_{ISO3}_FULL.csv
│           ├── building_enrichment_tsi_proportions_{ISO3}_RES_FULL.csv
│           ├── building_enrichment_tsi_proportions_{ISO3}_COM_FULL.csv
│           └── building_enrichment_tsi_proportions_{ISO3}_IND_FULL.csv
├── logs/
└── config.json (auto-generated by Task 0)
```

## Running for Multiple Countries

To process multiple countries:

1. In the notebook, change `ISO3 = "USA"` to `ISO3 = "GBR"` (for example)
2. Update input file paths if different per country
3. Re-run all cells

Each country gets its own:
- Folder structure: `{ISO3}/`
- Tables with suffix: `_<ISO3>`
- Export files: `FULL_{ISO3}/`
- Config file: `{ISO3}/config.json`

## Troubleshooting

### Package Installation Issues

If you see import errors:
1. The notebook auto-installs packages and restarts Python
2. After restart, re-run all cells from the beginning
3. Check that your cluster has internet access for PyPI

### Cluster Not Found

If cluster auto-detection fails:
1. Get your cluster ID from Databricks UI
2. In the notebook, set: `CLUSTER_ID = "1234-567890-abcd1234"`

### Table Does Not Exist Error

If exports fail with "table does not exist":
1. Check that all tasks completed successfully (check job logs)
2. Verify table name includes ISO3 suffix: `building_enrichment_output_{ISO3}`
3. Check Task 0 logs to ensure config was generated correctly

### File Copy Errors

If Task 0 fails with file copy errors:
1. Verify source paths exist and are accessible
2. Check you have read permissions on source paths
3. Check you have write permissions on volume paths
4. Ensure paths use correct format:
   - Workspace: `/Workspace/...`
   - Volumes: `/Volumes/...`
   - DBFS: `dbfs:/...`

### Task 0 Failures

If Task 0 (setup) fails:
1. Check Task 0 logs in Databricks UI
2. Verify `config_builder.py` exists in workspace
3. Verify input paths are correct
4. Check permissions on target volume

### Job Creation Failures

If job creation fails:
1. Verify you have permission to create jobs
2. Check that the cluster exists and is running
3. Verify workspace paths are correct
4. Check that all required files exist in workspace:
   - `task0_setup.py`
   - `task1-7.py`
   - `config_builder.py`
   - `requirements.txt`
   - `ghsl2_0_mwd_l1_tile_schema_land.gpkg`

## Advanced Configuration

### Custom Processing Parameters

In the notebook, you can edit these parameters in Cell 2:

```python
CELL_SIZE = 2000              # Grid cell size in meters
DOWNLOAD_CONCURRENCY = 3      # Parallel tile downloads
MAX_WORKERS = 8               # Raster processing threads
TILE_PARALLELISM = 4          # Concurrent tile processing
```

### Using a Different Cluster for Job Execution

By default, the job uses the same cluster as the notebook. To use a different cluster:

```python
CLUSTER_ID = "different-cluster-id-here"  # Specify instead of leaving empty
```

### Pipeline Tasks

The pipeline runs 8 tasks:
0. **Setup** - Create folders, copy files, generate config
1. Load proportions to Delta
2. Generate grid centroids
3. Download GHSL tiles
4. Extract raster statistics
5. Post-processing (join & enrich)
6. Create views
7. Export to CSV/Excel

All tasks are required for complete execution.

## Support

For issues or questions:
1. Check the job logs in Databricks UI (Workflows → Jobs → Building_Enrichment_{ISO3})
2. Review task-specific error messages (especially Task 0 for setup issues)
3. Verify all prerequisites are met
4. Check file paths and permissions in the notebook configuration

## Next Steps

After successful execution:
1. Review the exported CSV files in `{ISO3}/output/exports/FULL_{ISO3}/`
2. Validate the Excel summary
3. Query Delta tables directly for custom analysis
4. Inspect `{ISO3}/config.json` to see the full generated configuration
5. Set up scheduled runs if needed (manually in Databricks UI)

---

**Note**: This pipeline does not require Databricks CLI, Asset Bundles, config files, or command-line tools. Everything runs through the Jupyter notebook interface with configuration directly in the notebook cells.
