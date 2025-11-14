# Building Enrichment Pipeline - Quick Start Instructions

## Overview
This pipeline processes geospatial building data to generate enriched building inventories with TSI (Total Sum Insured) calculations and export capabilities. Everything runs via a single Jupyter notebook - **no CLI or config files required**.

**NEW**: All input files are in the `data/` folder! Just replace your NOS file and run.

## Prerequisites

1. **Databricks Workspace** with access to:
   - A cluster with Databricks Runtime 13.3+ LTS
   - Unity Catalog enabled
   - Volumes for data storage

2. **Input Data** (mostly included!):
   - ✅ TSI CSV (already in `data/tsi.csv`)
   - ✅ Admin boundaries (already in `data/RMS_Admin0_geozones.json.gz`)
   - ✅ Tile footprint (already in `data/ghsl2_0_mwd_l1_tile_schema_land.gpkg`)
   - 📝 **YOU PROVIDE**: Your country-specific NOS storey mapping CSV

3. **Permissions**:
   - Read/write access to specified catalog and schema
   - Read/write access to specified volumes
   - Ability to create and run jobs on your cluster

## Step-by-Step Instructions

### Step 1: Upload Files to Databricks Workspace

1. Upload the entire `mre/job1/` folder to your Databricks workspace
2. Recommended location: `/Workspace/Users/<your-email>/code-for-copilot/mre/job1/`
3. The `data/` folder contains all input files needed

### Step 2: Replace Your NOS File

**IMPORTANT**: Replace `data/NOS_storey_mapping.csv` with your country-specific file.

**Expected format** (see `data/NOS_storey_mapping_TEMPLATE.csv`):
```csv
NOS,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8
1,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0
3,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0
...
```

Where:
- `NOS`: Number of stories value
- `P_1` to `P_8`: Proportions for each storey (must sum to 1.0)

### Step 3: Open the Notebook

Open `create_and_run_job.ipynb` in Databricks and attach it to a cluster (Runtime 13.3+ LTS recommended)

### Step 4: Edit Configuration IN the Notebook

In **Cell 2** of the notebook, you only need to edit:

```python
# Country code (CHANGE THIS for your country)
ISO3 = "IND"  # Change to your country's ISO3 code

# Databricks settings (if different from defaults)
CATALOG = "prp_mr_bdap_projects"
SCHEMA = "geospatialsolutions"
VOLUME_BASE = "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data"

# Workspace path (where these scripts are located)
WORKSPACE_BASE = "/Workspace/Users/<your-email>/code-for-copilot/mre/job1"

# Optional: Email for notifications
EMAIL = "your-email@company.com"
```

**That's it!** Input files are automatically read from the `data/` folder:
```python
# These are already set in the notebook:
PROPORTIONS_CSV = f"{WORKSPACE_BASE}/data/NOS_storey_mapping.csv"
TSI_CSV = f"{WORKSPACE_BASE}/data/tsi.csv"
ADMIN_BOUNDARIES = f"{WORKSPACE_BASE}/data/RMS_Admin0_geozones.json.gz"
```

### Step 5: Run All Cells

Run all cells in the notebook. It will automatically:
- ✅ Auto-install required packages (databricks-sdk, pyyaml)
- ✅ Auto-detect cluster ID
- ✅ Generate minimal config using files from `data/` folder
- ✅ Create Databricks job with 8 tasks:
  - **Task 0**: Setup (creates folders, copies files, generates full config)
  - **Task 1-7**: Pipeline execution
- ✅ Run the job
- ✅ Monitor progress in real-time
- ✅ Verify outputs

### Step 6: What Happens in the Job

**Task 0 (Setup)** automatically:
1. Creates `{ISO3}/input/`, `{ISO3}/output/`, `{ISO3}/logs/` folders
2. Copies tiles from `data/` to `{ISO3}/input/tiles/`
3. Copies your NOS CSV from `data/` to `{ISO3}/input/`
4. Copies TSI and admin boundaries from `data/` to `{ISO3}/input/`
5. Uses `config_builder.py` to generate full `config.json` with ISO3 suffixes
6. Saves config to `{ISO3}/config.json`

**Task 1-7** (Pipeline):
- Read `{ISO3}/config.json`
- Execute the complete pipeline
- All tables created with `_{ISO3}` suffix

### Step 7: Monitor Progress

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

### Step 8: Access Your Results

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
{ISO3}/                          # e.g., IND/
├── input/
│   ├── tiles/
│   │   ├── ghsl2_0_mwd_l1_tile_schema_land.gpkg
│   │   ├── built_c/  (created during download)
│   │   └── smod/     (created during download)
│   ├── tsi.csv
│   ├── NOS_storey_mapping.csv
│   └── RMS_Admin0_geozones.json.gz
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

1. Replace `data/NOS_storey_mapping.csv` with the new country's file
2. In the notebook, change `ISO3 = "IND"` to `ISO3 = "USA"` (for example)
3. Re-run all cells

Each country gets its own:
- Folder structure: `{ISO3}/`
- Tables with suffix: `_<ISO3>`
- Export files: `FULL_{ISO3}/`
- Config file: `{ISO3}/config.json`

## Test Mode

For quick validation before running the full pipeline:

```python
# In Cell 2 of the notebook:
RUN_MODE = "test"  # Processes only 1 tile with 10k grid cells
```

This allows you to test the complete workflow in ~5 minutes instead of 45+ minutes.

Change to `RUN_MODE = "full"` for production runs.

## Troubleshooting

### Package Installation Issues

If you see import errors:
1. The notebook auto-installs packages and restarts Python
2. After restart, re-run all cells from the beginning
3. Check that your cluster has internet access for PyPI
4. Each job task now double-checks dependencies via `dependency_manager.py`. It
   installs `requirements.txt` on the attached cluster the first time a task
   runs (tracked via `/tmp/mre_job1_requirements.sha256`). Delete that file to
   force a reinstall if you change the requirements.

### Library Import Error

If you see `ImportError: cannot import name 'Library'`:
- This is fixed in the latest version
- Make sure you have the updated `job_creator.py` that imports from `databricks.sdk.service.compute`

### Cluster Not Found

If cluster auto-detection fails:
1. Get your cluster ID from Databricks UI
2. In the notebook, set: `CLUSTER_ID = "1234-567890-abcd1234"`

### NOS File Format Error

If Task 1 fails with proportion errors:
1. Verify your NOS CSV matches the expected format (see `data/NOS_storey_mapping_TEMPLATE.csv`)
2. Check that proportions (P_1 to P_8) sum to 1.0 for each NOS value
3. Ensure column names are exactly: `NOS,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8`

### Admin Boundaries File

The pipeline can read `.json.gz` files directly (compressed GeoJSON).
- GeoPandas reads `RMS_Admin0_geozones.json.gz` without manual decompression
- No need to extract the `.gz` file

### File Copy Errors

If Task 0 fails with file copy errors:
1. Verify `data/` folder contains all required files:
   - `NOS_storey_mapping.csv` (your file)
   - `tsi.csv`
   - `RMS_Admin0_geozones.json.gz`
   - `ghsl2_0_mwd_l1_tile_schema_land.gpkg`
2. Check workspace path is correct in Cell 2
3. Verify you have read permissions on workspace files
4. Check you have write permissions on volume paths

### Task 0 Failures

If Task 0 (setup) fails:
1. Check Task 0 logs in Databricks UI
2. Verify all files exist in `data/` folder
3. Verify `config_builder.py` exists in workspace
4. Check permissions on target volume

### Job Creation Failures

If job creation fails:
1. Verify you have permission to create jobs
2. Check that the cluster exists and is running
3. Verify workspace paths are correct
4. Check that all required files exist:
   - `data/` folder with input files
   - `task0_setup.py` through `task7_export.py`
   - `config_builder.py`, `config_generator.py`, `job_creator.py`, `job_monitor.py`
   - `requirements.txt`

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

## What's in the data/ Folder

```
data/
├── README.md                              # This explains the folder
├── NOS_storey_mapping.csv                 # ⚠️ YOU MUST PROVIDE THIS
├── NOS_storey_mapping_TEMPLATE.csv        # Template showing expected format
├── tsi.csv                                # ✅ Provided
├── RMS_Admin0_geozones.json.gz           # ✅ Provided (compressed GeoJSON)
└── ghsl2_0_mwd_l1_tile_schema_land.gpkg  # ✅ Provided
```

## Support

For issues or questions:
1. Check the job logs in Databricks UI (Workflows → Jobs → Building_Enrichment_{ISO3})
2. Review task-specific error messages (especially Task 0 for setup issues)
3. Verify your NOS file format matches the template
4. Check that all files are present in the `data/` folder

## Next Steps

After successful execution:
1. Review the exported CSV files in `{ISO3}/output/exports/FULL_{ISO3}/`
2. Validate the Excel summary
3. Query Delta tables directly for custom analysis
4. Inspect `{ISO3}/config.json` to see the full generated configuration
5. Set up scheduled runs if needed (manually in Databricks UI)

---

**Note**: This pipeline does not require Databricks CLI, Asset Bundles, external config files, or command-line tools. Everything runs through the Jupyter notebook interface with configuration directly in the notebook cells, and all input files come from the `data/` folder.
