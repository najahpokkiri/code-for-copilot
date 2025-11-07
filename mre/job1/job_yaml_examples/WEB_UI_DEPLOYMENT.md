# Deploying to Databricks Web UI (No CLI Access)

## 🎯 Your Situation

- ✅ Have: Databricks Web UI access
- ✅ Have: Git repository with all configs and scripts
- ❌ Don't have: Command-line access / Databricks CLI
- ❌ Don't have: Local terminal to run `databricks bundle deploy`

**Solution**: Use Databricks Web UI features to deploy everything!

---

## 🚀 Option 1: Databricks Repos (RECOMMENDED)

Use Databricks' built-in Git integration to connect your repository.

### Step 1: Connect Your Git Repository

1. **Open Databricks Web UI**
2. **Navigate to**: Workspace → Repos
3. **Click**: "Add Repo"
4. **Fill in**:
   - Git URL: `https://github.com/najahpokkiri/code-for-copilot`
   - Git provider: GitHub
   - Branch: `claude/job-yaml-config-structure-011CUthFZWYLsccLNE2WRN5c`
   - Repo name: `code-for-copilot`

5. **Click**: "Create Repo"

**Result**: Your entire Git repository is now synced to Databricks!

### Step 2: Locate Your Files

After repo is connected:

```
Workspace → Repos → code-for-copilot → mre → job1
├── config.yaml ✅
├── config_builder.py ✅
├── task1_proportions_to_delta.py ✅
├── task2_grid_generation.py ✅
├── ... (all scripts)
```

### Step 3: Upload Input Data Files

Your scripts are in Repos, but **input data files** need to be uploaded separately:

#### Option A: Upload to Workspace

1. **Navigate to**: Workspace → Users → your_email@company.com
2. **Create folder**: `inventory_nos_db/data/`
3. **Click**: Upload
4. **Upload your files**:
   - `IND_NOS_storey_mapping.csv`
   - `tsi.csv` (or keep in Volumes)

#### Option B: Upload to Volumes (RECOMMENDED)

1. **Navigate to**: Catalog → Volumes
2. **Find your volume**: `prp_mr_bdap_projects.geospatialsolutions.external`
3. **Navigate to**: `jrc/data/inputs/`
4. **Create folders** (if not exist):
   - `proportions/`
   - `multipliers/`
   - `admin/`
   - `tiles/`

5. **Upload files** to appropriate folders:
   - `IND_NOS_storey_mapping.csv` → `proportions/`
   - `tsi.csv` → `multipliers/`
   - `RMS_Admin0_geozones.gpkg` → `admin/` (250 MB - already there?)
   - `GHSL_tile_footprint.gpkg` → `tiles/`

### Step 4: Create the Job in Web UI

Now create the Databricks Job using the UI:

#### Method A: Import Job from YAML (If Supported)

1. **Navigate to**: Workflows → Jobs
2. **Click**: "Create Job" or "Import"
3. **If Import is available**:
   - Upload: `job_yaml_examples/databricks_bundle_example/resources/jobs/building_enrichment.yml`
   - Adjust paths to point to Repos

#### Method B: Create Job Manually (Most Common)

1. **Navigate to**: Workflows → Jobs
2. **Click**: "Create Job"
3. **Job name**: `Building Data Enrichment - IND`

4. **Add Task 0** (Config Generation):
   - Task name: `task0_config_generation`
   - Type: Python
   - Source: Workspace file
   - Python file path: `/Repos/code-for-copilot/mre/job1/config_builder.py`
   - Parameters: `["config.yaml"]`
   - Cluster: Select existing cluster or create new

5. **Add Task 1** (Import Proportions):
   - Task name: `task1_import_proportions`
   - Type: Python
   - Source: Workspace file
   - Python file path: `/Repos/code-for-copilot/mre/job1/task1_proportions_to_delta.py`
   - Parameters:
     ```
     ["--config_path", "/Repos/code-for-copilot/mre/job1/config.json"]
     ```
   - Depends on: `task0_config_generation`
   - Cluster: Same as Task 0

6. **Repeat for Tasks 2-7** following the pattern from the job YAML

**Task dependency structure**:
```
Task 0 (config_generation)
    ↓
Task 1 (import_proportions)
    ↓
Task 2 (grid_generation) - Needs libraries: geopandas, shapely
    ↓
Task 3 (tile_download)
    ↓
Task 4 (raster_stats) - Needs libraries: rasterio, geopandas
    ↓
Task 5 (post_processing)
    ↓
Task 6 (create_views)
    ↓
Task 7 (export) - Needs library: xlsxwriter
```

### Step 5: Configure Cluster

For each task (or use shared cluster):

1. **Cluster settings**:
   - Runtime: DBR 13.3 LTS or newer
   - Node type: Standard_DS3_v2 (or available)
   - Workers: 4
   - Libraries:
     - Task 2: `geopandas==0.14.4`, `shapely==2.0.4`
     - Task 4: `rasterio==1.3.9`
     - Task 7: `xlsxwriter==3.2.9`

2. **To add libraries**:
   - In job task configuration
   - Go to "Libraries" section
   - Click "Add" → PyPI
   - Enter package name and version

### Step 6: Update Config Paths for Repos

Edit `config.yaml` in your Repo to use Repos paths:

**Before** (if you had workspace paths):
```yaml
inputs:
  proportions_csv: /Workspace/Users/.../data/IND_NOS_storey_mapping.csv
```

**After** (using Volumes - RECOMMENDED):
```yaml
inputs:
  proportions_csv: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/proportions/IND_NOS_storey_mapping.csv
  tsi_csv: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/multipliers/tsi.csv
  admin_boundaries: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg
  tile_footprint: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.gpkg
```

**Note**: Task scripts are in `/Repos/...` but data files should be in `/Volumes/...`

### Step 7: Run the Job

1. **Navigate to**: Workflows → Jobs → Your Job
2. **Click**: "Run Now"
3. **Monitor**: Watch task progress in the UI
4. **Check logs**: Click on each task to see logs

---

## 🚀 Option 2: Upload Scripts to Workspace

If you can't use Repos, upload scripts manually:

### Step 1: Export Scripts from Git

Download from your Git repository:
- All `task*.py` files
- `config_builder.py`
- `config.yaml`

### Step 2: Upload to Workspace

1. **Navigate to**: Workspace → Users → your_email@company.com
2. **Create folder**: `inventory_nos_db/scripts/`
3. **Upload all Python files**
4. **Upload config.yaml**

### Step 3: Create Job (Same as Option 1, Method B)

But use Workspace paths:
```
Python file path: /Workspace/Users/your_email@company.com/inventory_nos_db/scripts/task1_proportions_to_delta.py
```

---

## 🚀 Option 3: Use Notebooks Instead of .py Files

Convert your Python scripts to notebooks:

### Step 1: Create Notebooks in Web UI

1. **Navigate to**: Workspace → Users → your_email
2. **Create notebook**: "Task 0 - Config Generation"
3. **In first cell**, paste content of `config_builder.py`
4. **Add parameters cell**:
   ```python
   dbutils.widgets.text("config_path", "config.yaml", "Config Path")
   config_path = dbutils.widgets.get("config_path")
   ```

5. **Repeat for all tasks**

### Step 2: Create Job with Notebook Tasks

1. **Navigate to**: Workflows → Jobs → Create Job
2. **Add tasks** with type: Notebook
3. **Notebook path**: `/Users/your_email/Task0_Config_Generation`
4. **Parameters**: `{"config_path": "config.yaml"}`

---

## 📊 Comparison: Which Option?

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Repos** ⭐ | Auto-sync with Git, Easy updates, Version control | Need Git access | Teams, CI/CD |
| **Workspace Upload** | Simple, No Git needed | Manual updates, No version control | One-off runs |
| **Notebooks** | Native Databricks, Easy to debug | More work to convert, Harder to version | Interactive development |

**Recommendation**: Use **Option 1 (Repos)** - it's the most maintainable!

---

## 🗂️ File Locations Summary

After setup with Repos:

```
SCRIPTS (in Repos):
/Repos/code-for-copilot/mre/job1/
├── config.yaml ← Edit here
├── config_builder.py
├── task1_proportions_to_delta.py
├── task2_grid_generation.py
└── ... (all task scripts)

DATA FILES (in Volumes):
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/
├── inputs/
│   ├── proportions/
│   │   └── IND_NOS_storey_mapping.csv ← Upload via UI
│   ├── multipliers/
│   │   └── tsi.csv ← Upload via UI
│   ├── admin/
│   │   └── RMS_Admin0_geozones.gpkg ← Upload via UI (250 MB)
│   └── tiles/
│       └── GHSL_tile_footprint.gpkg ← Upload via UI
└── outputs/
    └── exports/ ← Generated by pipeline
```

---

## ✅ Quick Start Checklist (Using Repos)

- [ ] 1. Connect Git repo to Databricks Repos
- [ ] 2. Upload input data files to Volumes
- [ ] 3. Verify config.yaml paths point to Volumes
- [ ] 4. Create Job in Web UI with 8 tasks (0-7)
- [ ] 5. Set task dependencies (0→1→2→3→4→5→6→7)
- [ ] 6. Configure cluster and libraries for each task
- [ ] 7. Run job and monitor progress

---

## 🎓 Key Differences: CLI vs Web UI

| Aspect | With CLI | With Web UI Only |
|--------|----------|------------------|
| **Deploy bundle** | `databricks bundle deploy` | Connect Repos OR Upload files |
| **Create job** | Auto-created from YAML | Manual creation in UI |
| **Upload data** | Auto-synced | Manual upload to Volumes |
| **Update scripts** | Push to Git → redeploy | Push to Git → Repos auto-sync |
| **Configuration** | Single command | Multiple UI steps |

**The good news**: Using Repos gives you most CLI benefits through the UI!

---

## 💡 Pro Tips for Web UI Workflow

1. **Use Repos for scripts**: Auto-syncs with Git when you push changes
2. **Use Volumes for data**: Large files, easy to manage
3. **Save job as template**: After creating job once, export/clone for other countries
4. **Use job parameters**: Make ISO3 a job parameter to switch countries easily
5. **Enable email notifications**: Get alerts when job completes

---

## 📖 Next Steps

1. **Start with Option 1 (Repos)** - Connect your Git repository
2. **Upload data files** to appropriate Volumes locations
3. **Create the job manually** in Web UI (first time)
4. **Save job configuration** for reuse
5. **Run and test** with your IND data

**All the YAML files and documentation we created are still valuable as references** for:
- Understanding the job structure
- Knowing what parameters each task needs
- Seeing task dependencies
- Library requirements

You just configure them through the UI instead of CLI!

---

## ❓ Questions?

See the Databricks documentation:
- [Databricks Repos](https://docs.databricks.com/repos)
- [Creating Jobs](https://docs.databricks.com/workflows/jobs/create-jobs)
- [Using Volumes](https://docs.databricks.com/connect/unity-catalog/volumes)
