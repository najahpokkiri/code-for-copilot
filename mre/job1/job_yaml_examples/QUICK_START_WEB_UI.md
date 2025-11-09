# Quick Start: Databricks Web UI (No CLI)

## 🎯 Goal

Deploy your pipeline using **only Databricks Web UI** (no command line needed).

---

## ⚡ 3 Simple Steps

### Step 1: Connect Your Git Repo (5 minutes)

**In Databricks Web UI:**

1. Click **Workspace** (left sidebar)
2. Click **Repos**
3. Click **Add Repo** button
4. Fill in:
   - **Git URL**: `https://github.com/najahpokkiri/code-for-copilot`
   - **Branch**: `claude/job-yaml-config-structure-011CUthFZWYLsccLNE2WRN5c`
   - **Repo name**: `code-for-copilot`
5. Click **Create Repo**

**✅ Result**: Your scripts are now at `/Repos/code-for-copilot/mre/job1/`

---

### Step 2: Upload Your Data Files (10 minutes)

**In Databricks Web UI:**

1. Click **Catalog** (left sidebar)
2. Navigate to your volume: `prp_mr_bdap_projects` → `geospatialsolutions` → `external` → Volumes → `jrc`
3. Navigate to: `data/inputs/`

4. **Create folders** (if they don't exist):
   - `proportions/`
   - `multipliers/`
   - `admin/`
   - `tiles/`

5. **Upload your files**:
   - Go to `proportions/` → Click **Upload** → Select `IND_NOS_storey_mapping.csv`
   - Go to `multipliers/` → Click **Upload** → Select `tsi.csv`
   - Go to `admin/` → Upload `RMS_Admin0_geozones.gpkg` (if not already there)
   - Go to `tiles/` → Upload `GHSL_tile_footprint.gpkg`

**✅ Result**: Data files are in Volumes at:
```
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/
├── proportions/IND_NOS_storey_mapping.csv
├── multipliers/tsi.csv
├── admin/RMS_Admin0_geozones.gpkg
└── tiles/GHSL_tile_footprint.gpkg
```

---

### Step 3: Create the Job (20 minutes)

**In Databricks Web UI:**

1. Click **Workflows** (left sidebar)
2. Click **Jobs**
3. Click **Create Job**
4. Name it: `Building Data Enrichment - IND`

#### Add Task 0: Config Generation

- **Task name**: `task0_config_generation`
- **Type**: Python script
- **Source**: Workspace
- **Path**: `/Repos/code-for-copilot/mre/job1/config_builder.py`
- **Parameters** (click "Add" → "Positional"):
  ```
  config.yaml
  ```
- **Cluster**: Select existing cluster (or create new with DBR 13.3+)
- Click **Create**

#### Add Task 1: Import Proportions

- Click **Add task**
- **Task name**: `task1_import_proportions`
- **Type**: Python script
- **Source**: Workspace
- **Path**: `/Repos/code-for-copilot/mre/job1/task1_proportions_to_delta.py`
- **Parameters** (click "Add" → "Named"):
  - Key: `--config_path`
  - Value: `/Repos/code-for-copilot/mre/job1/config.json`
- **Depends on**: Select `task0_config_generation`
- **Cluster**: Same as Task 0
- Click **Create**

#### Add Task 2: Grid Generation

- Click **Add task**
- **Task name**: `task2_grid_generation`
- **Type**: Python script
- **Path**: `/Repos/code-for-copilot/mre/job1/task2_grid_generation.py`
- **Parameters**:
  - Key: `--config_path`
  - Value: `/Repos/code-for-copilot/mre/job1/config.json`
- **Depends on**: `task1_import_proportions`
- **Libraries** (click "Add library" → PyPI):
  - `geopandas==0.14.4`
  - `shapely==2.0.4`
- Click **Create**

#### Add Task 3: Tile Download

- **Task name**: `task3_tile_download`
- **Path**: `/Repos/code-for-copilot/mre/job1/task3_tile_downloader.py`
- **Parameters**: Same as Task 2
- **Depends on**: `task2_grid_generation`

#### Add Task 4: Raster Stats

- **Task name**: `task4_raster_stats`
- **Path**: `/Repos/code-for-copilot/mre/job1/task4_raster_stats.py`
- **Parameters**: Same as Task 2
- **Depends on**: `task3_tile_download`
- **Libraries**:
  - `rasterio==1.3.9`
  - `geopandas==0.14.4`

#### Add Task 5: Post-Processing

- **Task name**: `task5_post_processing`
- **Path**: `/Repos/code-for-copilot/mre/job1/task5_post_processing.py`
- **Parameters**: Same as Task 2
- **Depends on**: `task4_raster_stats`

#### Add Task 6: Create Views

- **Task name**: `task6_create_views`
- **Path**: `/Repos/code-for-copilot/mre/job1/task6_create_views.py`
- **Parameters**: Same as Task 2
- **Depends on**: `task5_post_processing`

#### Add Task 7: Export

- **Task name**: `task7_export`
- **Path**: `/Repos/code-for-copilot/mre/job1/task7_export.py`
- **Parameters**: Same as Task 2
- **Depends on**: `task6_create_views`
- **Libraries**:
  - `xlsxwriter==3.2.9`

**✅ Result**: Job with 8 tasks (0-7) ready to run!

---

## ▶️ Run the Job

1. In your job page, click **Run Now**
2. Watch the progress:
   - Task 0 runs first (generates config.json)
   - Task 1 runs after Task 0 completes
   - And so on...

3. **Check logs**: Click on each task to see detailed logs
4. **Check outputs**: Navigate to Catalog → Tables to see created Delta tables

---

## 📋 Visual Task Flow

```
┌─────────────────────────────────┐
│ Task 0: Config Generation       │
│ Creates config.json from YAML   │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 1: Import Proportions      │
│ Loads CSV to Delta              │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 2: Grid Generation         │
│ Creates grid cells              │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 3: Tile Download           │
│ Downloads GHSL tiles            │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 4: Raster Stats            │
│ Extracts building counts        │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 5: Post-Processing         │
│ Calculates estimates            │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 6: Create Views            │
│ Generates SQL views             │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Task 7: Export                  │
│ Exports final results           │
└─────────────────────────────────┘
```

---

## 🔍 Verify Your Setup

Before running, check:

- [ ] Repos connected: `/Repos/code-for-copilot/` exists
- [ ] Data uploaded: Check Volumes → `jrc/data/inputs/`
- [ ] Job created: 8 tasks with correct dependencies
- [ ] Libraries added: Task 2 (geopandas), Task 4 (rasterio), Task 7 (xlsxwriter)

---

## 💡 Pro Tips

1. **Clone for other countries**: After creating job once, click "Clone" to create USA, BRA versions
2. **Save cluster config**: Use job cluster definition for consistent runs
3. **Enable notifications**: Add your email in job settings
4. **Use job parameters**: Make ISO3 a parameter for easy country switching

---

## ❓ Troubleshooting

### Issue: "File not found" in Task 0

**Solution**: Check that `config.yaml` exists at `/Repos/code-for-copilot/mre/job1/config.yaml`

### Issue: "Module not found: geopandas"

**Solution**: Add library to task: Click task → Libraries → Add → PyPI → `geopandas==0.14.4`

### Issue: Task 1+ can't find config.json

**Solution**: Verify Task 0 completed successfully and generated config.json in the Repos folder

### Issue: Large file upload fails

**Solution**: For files > 100 MB, use Databricks CLI or dbutils (in notebook):
```python
# In a notebook:
dbutils.fs.cp("file:/path/to/large/file.gpkg",
              "dbfs:/Volumes/catalog/schema/volume/path/")
```

---

## 📖 What About All Those YAML Files?

The job YAML files we created are **reference documentation** showing:
- What the job structure should be
- Task dependencies
- Required libraries
- Configuration settings

You use them as a **guide** when creating the job in the Web UI.

---

## 🎉 Success!

Once job completes successfully:
- Check output tables in Catalog
- View generated exports in Volumes
- Review task logs for any warnings

---

**Need detailed guide?** See [WEB_UI_DEPLOYMENT.md](./WEB_UI_DEPLOYMENT.md)
