# Building Enrichment Pipeline - Quick Start Instructions

## Overview
This pipeline processes geospatial building data to generate enriched building inventories with TSI (Total Sum Insured) calculations and export capabilities. Everything runs via a single Jupyter notebook - **no CLI required**.

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

### Step 2: Configure Your Job

Edit the `job_config.yaml` file with your settings:

```yaml
# Set your country code
iso3: "USA"  # Change to your country's ISO3 code

# Update input file paths
inputs:
  proportions_csv: "/path/to/your/proportions.csv"
  tsi_csv: "/path/to/your/tsi.csv"
  world_boundaries: "/path/to/your/world.gpkg"  # Optional

# Update Databricks settings
databricks:
  catalog: "your_catalog"
  schema: "your_schema"
  workspace_base: "/Workspace/Users/<your-email>/code-for-copilot/mre/job1"
  volume_base: "/Volumes/your_catalog/your_schema/external/data"
  cluster_id: ""  # Leave empty to auto-detect

# Optional: Add your email for notifications
job:
  email_notifications: "your-email@company.com"
```

### Step 3: Open and Run the Notebook

1. Open `create_and_run_job.ipynb` in Databricks
2. Attach it to a cluster (Runtime 13.3+ LTS recommended)
3. Run all cells sequentially

The notebook will:
- ✅ Auto-install required packages
- ✅ Create `{ISO3}/input/`, `{ISO3}/output/`, `{ISO3}/logs/` folders
- ✅ Copy input files to correct locations
- ✅ Copy tiles to `{ISO3}/input/tiles/`
- ✅ Generate complete configuration with ISO3 suffixes
- ✅ Create Databricks job with all tasks
- ✅ Run the job
- ✅ Monitor progress in real-time
- ✅ Verify outputs

### Step 4: Monitor Progress

The notebook displays real-time job progress:

```
⏳ Monitoring job progress...

[0s] Job status: PENDING
[30s] Job status: RUNNING
[35s] ⏳ task1_proportions_to_delta: RUNNING
[180s] ✅ task1_proportions_to_delta: TERMINATED
[185s] ⏳ task2_grid_generation: RUNNING
...
✅ Job completed successfully!
   Duration: 45m 23s
```

### Step 5: Access Your Results

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

## Folder Structure Created

```
{ISO3}/                          # e.g., USA/
├── input/
│   ├── tiles/
│   │   └── ghsl2_0_mwd_l1_tile_schema_land.gpkg
│   ├── tsi.csv
│   ├── proportions.csv
│   └── world_boundaries.gpkg (optional)
├── output/
│   └── exports/
│       └── FULL_{ISO3}/
│           ├── building_enrichment_output_{ISO3}_FULL.csv
│           ├── building_enrichment_tsi_proportions_{ISO3}_RES_FULL.csv
│           ├── building_enrichment_tsi_proportions_{ISO3}_COM_FULL.csv
│           └── building_enrichment_tsi_proportions_{ISO3}_IND_FULL.csv
├── logs/
└── config.json (auto-generated)
```

## Running for Multiple Countries

To process multiple countries:

1. Update `iso3` in `job_config.yaml`
2. Update input file paths if different per country
3. Re-run the notebook

Each country gets its own:
- Folder structure: `{ISO3}/`
- Tables with suffix: `_<ISO3>`
- Export files: `FULL_{ISO3}/`

## Troubleshooting

### Package Installation Issues

If you see import errors:
1. The notebook auto-installs packages and restarts Python
2. After restart, re-run all cells from the beginning
3. Check that your cluster has internet access for PyPI

### Cluster Not Found

If cluster auto-detection fails:
1. Get your cluster ID from Databricks UI
2. Add it to `job_config.yaml`:
   ```yaml
   databricks:
     cluster_id: "1234-567890-abcd1234"
   ```

### Table Does Not Exist Error

If exports fail with "table does not exist":
1. Check that all tasks completed successfully
2. Verify table name includes ISO3 suffix: `building_enrichment_output_{ISO3}`
3. Check the job logs in Databricks UI for task failures

### File Copy Errors

If file copies fail:
1. Verify source paths exist and are accessible
2. Check you have read permissions on source paths
3. Check you have write permissions on volume paths
4. Ensure paths use correct format:
   - Workspace: `/Workspace/...`
   - Volumes: `/Volumes/...`
   - DBFS: `dbfs:/...`

### Job Creation Failures

If job creation fails:
1. Verify you have permission to create jobs
2. Check that the cluster exists and is running
3. Verify workspace paths are correct
4. Check that `requirements.txt` exists in workspace

## Advanced Configuration

### Custom Processing Parameters

Edit `params` section in `job_config.yaml`:

```yaml
params:
  cell_size: 2000                # Grid cell size in meters
  download_concurrency: 3        # Parallel downloads
  max_workers: 8                 # Parallel processing workers
```

### Using a Different Cluster for Job Execution

By default, the job uses the same cluster as the notebook. To use a different cluster:

```yaml
databricks:
  cluster_id: "different-cluster-id-here"
```

### Skipping Optional Steps

The pipeline runs all 7 tasks by default. Tasks are:
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
1. Check the job logs in Databricks UI (Workflows → Jobs → Your Job Name)
2. Review task-specific error messages
3. Verify all prerequisites are met
4. Check file paths and permissions

## Next Steps

After successful execution:
1. Review the exported CSV files
2. Validate the Excel summary
3. Query Delta tables directly for custom analysis
4. Set up scheduled runs if needed (manually in Databricks UI)

---

**Note**: This pipeline does not require Databricks CLI, Asset Bundles, or command-line tools. Everything runs through the Jupyter notebook interface.
