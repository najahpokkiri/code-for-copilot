# Setup Guide - Git Clone to Running Pipeline

**Complete guide for setting up the geospatial pipeline after cloning from git.**

---

## 📋 Overview

This guide walks you through:
1. Cloning the repository
2. Setting up for your environment
3. Deploying the Databricks job
4. Running the pipeline

**Estimated time:** 15-30 minutes

---

## 🎯 Prerequisites

### Required
- ✅ Databricks Workspace access
- ✅ Databricks cluster (or ability to create one)
- ✅ Git installed locally
- ✅ Python 3.8+ (for local testing)

### Optional but Recommended
- ✅ Databricks CLI installed
- ✅ PyYAML installed: `pip install pyyaml`

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Clone and setup
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline/mre/job1
cp config.yaml my_config.yaml
vim my_config.yaml  # Edit for your country

# 2. Generate config
python config_builder.py my_config.yaml

# 3. Deploy and run (see detailed steps below)
```

---

## 📖 Detailed Setup

### Step 1: Clone Repository

```bash
# Clone the repo
git clone https://github.com/your-org/geospatial-pipeline.git

# Navigate to job folder
cd geospatial-pipeline/mre/job1

# List files
ls -la
```

**Expected files:**
```
config.yaml              # Simplified configuration (EDIT THIS)
config_builder.py        # Config generator
task0_generate_config.py # Task 0 (auto-generates config)
task1_proportions_to_delta.py
task2_grid_generation.py
...
databricks_jobs/         # Job definitions folder
```

---

### Step 2: Configure for Your Country

#### 2.1 Copy Template

```bash
# Create your config from template
cp config.yaml config_MY_COUNTRY.yaml
```

#### 2.2 Edit Configuration

```bash
vim config_MY_COUNTRY.yaml
```

**Edit these sections:**

```yaml
# ============================================================================
# PROJECT SETTINGS (Update for your Databricks environment)
# ============================================================================
project:
  catalog: YOUR_CATALOG              # Your Databricks catalog
  schema: YOUR_SCHEMA                # Your Databricks schema
  volume_root: /Volumes/YOUR_PATH    # Your volume root path

# ============================================================================
# COUNTRY SETTINGS (Main thing to change!)
# ============================================================================
country:
  iso3: USA  # Change to your country (USA, BRA, GBR, etc.)

# ============================================================================
# INPUT FILES (Update paths to your data)
# ============================================================================
inputs:
  # Proportions CSV - building storey distribution
  proportions_csv: /path/to/YOUR_COUNTRY_proportions.csv

  # TSI CSV - Total Sum Insured multipliers
  tsi_csv: /path/to/tsi.csv

  # Admin boundaries - country boundaries GeoPackage
  admin_boundaries: /path/to/admin_boundaries.gpkg

  # Tile footprint - GHSL tile schema shapefile
  tile_footprint: /path/to/GHSL_tile_schema.shp

# ============================================================================
# PROCESSING PARAMETERS (Optional - tune for performance)
# ============================================================================
params:
  cell_size: 2000           # Keep at 2000m for 2km grids
  max_workers: 8            # Adjust based on cluster size
  chunk_size: 10000         # Adjust based on memory
  # ... rest can stay as defaults
```

#### 2.3 Generate Full Config

```bash
# Generate config.json from your YAML
python config_builder.py config_MY_COUNTRY.yaml --output config.json

# Verify it worked
cat config.json | head -20
```

**Expected output:**
```
✅ Generated configuration written to: config.json
✅ Configurations validated

CONFIGURATION SUMMARY
=====================
Catalog: YOUR_CATALOG
Schema: YOUR_SCHEMA
Country: USA
...
```

---

### Step 3: Upload to Databricks Workspace

#### Option A: Via Databricks CLI (Recommended)

```bash
# Authenticate with Databricks
databricks auth login --host https://your-workspace.cloud.databricks.com

# Create directory in Workspace
databricks workspace mkdirs /Workspace/Users/YOUR_EMAIL/geospatial_pipeline

# Upload files
databricks workspace import-dir \
  . \
  /Workspace/Users/YOUR_EMAIL/geospatial_pipeline \
  --overwrite

# Verify upload
databricks workspace ls /Workspace/Users/YOUR_EMAIL/geospatial_pipeline
```

#### Option B: Via Git Integration

```bash
# In Databricks UI:
# 1. Go to Repos
# 2. Click "Add Repo"
# 3. Enter your git URL
# 4. Click "Create Repo"

# Files automatically sync!
```

#### Option C: Manual Upload via UI

1. Go to Databricks Workspace
2. Navigate to your user folder
3. Right-click → Import
4. Upload each file manually

---

### Step 4: Setup Databricks Job

Choose one of the 3 approaches (see `databricks_jobs/README.md` for details):

#### ⭐ Recommended: Approach 2 (Parameterized)

**Edit job definition:**

```bash
vim databricks_jobs/approach2_parameterized.yml
```

**Update parameters:**

```yaml
parameters:
  - name: workspace_path
    default: /Workspace/Users/YOUR_EMAIL/geospatial_pipeline

  - name: iso3
    default: USA  # Your country

  - name: cluster_id
    default: YOUR_CLUSTER_ID  # Your cluster ID
```

**Deploy:**

```bash
# Via CLI
databricks jobs create --json-file databricks_jobs/approach2_parameterized.yml

# Returns job ID, e.g., 12345
```

**Or via UI:**
1. Go to Workflows → Create Job
2. Click "JSON" tab
3. Paste contents of `approach2_parameterized.yml`
4. Click "Create"

---

### Step 5: Run the Pipeline

#### Via CLI

```bash
# Run job
databricks jobs run-now --job-id YOUR_JOB_ID

# Monitor run
databricks runs get --run-id RETURNED_RUN_ID
```

#### Via UI

1. Go to Workflows
2. Find your job "Geospatial Pipeline"
3. Click "Run now"
4. Monitor progress in the Runs tab

---

## 🔍 Verify Results

### Check Task 0 Output

Task 0 should have auto-generated `config.json`:

```bash
# Via CLI
databricks workspace export /Workspace/Users/YOUR_EMAIL/geospatial_pipeline/config.json

# Should show the full 50+ line config
```

### Check Delta Tables

After pipeline completes, verify tables exist:

```sql
-- In Databricks SQL
SHOW TABLES IN YOUR_CATALOG.YOUR_SCHEMA;

-- Should see:
-- building_enrichment_proportions_input
-- building_enrichment_tsi_input
-- grid_centroids
-- download_status
-- grid_counts
-- building_enrichment_output
```

### Check Exports

```bash
# Via CLI
databricks fs ls /Volumes/YOUR_PATH/outputs/exports/

# Should see:
# FULL_USA/
# building_summary_country_layout_USA.xlsx
```

---

## 🐛 Troubleshooting

### Issue: "config.yaml not found"

**Solution:**
```bash
# Make sure you're in the right directory
cd mre/job1
ls config.yaml  # Should exist

# If you renamed it
python config_builder.py config_MY_COUNTRY.yaml
```

### Issue: "Missing required config keys"

**Solution:**
Check that your `config.yaml` has all required sections:
- `project` (catalog, schema, volume_root)
- `country` (iso3)
- `inputs` (all 4 file paths)
- `params` (processing parameters)

### Issue: Task 0 fails in Databricks

**Solution:**
```bash
# Make sure config_builder.py is in the same directory as task0
databricks workspace ls /Workspace/Users/YOUR_EMAIL/geospatial_pipeline/

# Should see both:
# config_builder.py
# task0_generate_config.py
```

### Issue: "Cluster ID not found"

**Solution:**
```bash
# List your clusters
databricks clusters list

# Use the cluster ID from output
# Update in job definition
```

### Issue: Permission denied on volumes

**Solution:**
```sql
-- In Databricks SQL, grant permissions
GRANT ALL PRIVILEGES ON VOLUME YOUR_CATALOG.YOUR_SCHEMA.YOUR_VOLUME TO `YOUR_EMAIL`;
```

---

## 📂 Directory Structure After Setup

```
geospatial-pipeline/
└── mre/
    └── job1/
        ├── config.yaml                    # Template (don't edit)
        ├── config_MY_COUNTRY.yaml         # Your config (edit this)
        ├── config.json                    # Auto-generated
        ├── config_builder.py              # Generator script
        ├── task0_generate_config.py       # Task 0
        ├── task1_proportions_to_delta.py  # Task 1
        ├── task2_grid_generation.py       # Task 2
        ├── ...                            # Tasks 3-7
        ├── databricks_jobs/
        │   ├── README.md                  # Job deployment guide
        │   ├── approach1_simple.yml       # Simple approach
        │   ├── approach2_parameterized.yml # Recommended
        │   └── approach3_asset_bundles.yml # Enterprise
        ├── PIPELINE_GUIDE.md              # Pipeline documentation
        ├── CONFIG_GUIDE.md                # Config documentation
        └── SETUP_GUIDE.md                 # This file
```

---

## 🎓 Tutorial: Complete Example (India)

Let's walk through a complete example for India:

### 1. Clone and Navigate

```bash
git clone https://github.com/your-org/geospatial-pipeline.git
cd geospatial-pipeline/mre/job1
```

### 2. Create Config

```bash
cp config.yaml config_india.yaml
vim config_india.yaml
```

**Edit:**
```yaml
country:
  iso3: IND

inputs:
  proportions_csv: /Volumes/.../IND_proportions.csv
  tsi_csv: /Volumes/.../tsi.csv
  admin_boundaries: /Volumes/.../admin/RMS_Admin0_geozones.gpkg
  tile_footprint: /Volumes/.../tiles/GHSL_tile_schema.shp
```

### 3. Generate and Validate

```bash
python config_builder.py config_india.yaml --validate
```

### 4. Upload to Databricks

```bash
databricks auth login
databricks workspace import-dir . /Workspace/Users/you/geo_india --overwrite
```

### 5. Create Job

```bash
# Edit job file
vim databricks_jobs/approach2_parameterized.yml

# Change:
#   workspace_path: /Workspace/Users/you/geo_india
#   iso3: IND
#   cluster_id: your-cluster-id

# Deploy
databricks jobs create --json-file databricks_jobs/approach2_parameterized.yml
# Returns: Created job with ID: 789
```

### 6. Run Pipeline

```bash
databricks jobs run-now --job-id 789

# Monitor
databricks runs list --job-id 789
```

### 7. Check Results

```bash
# List exports
databricks fs ls /Volumes/.../outputs/exports/FULL_IND/

# Download Excel summary
databricks fs cp \
  /Volumes/.../outputs/exports/building_summary_country_layout_IND.xlsx \
  ./results_india.xlsx
```

**Done! You've successfully set up and run the pipeline for India!**

---

## 🔄 Updating Configuration

### To Change Country

```bash
# Edit YAML
vim config_MY_COUNTRY.yaml
# Change iso3: USA → iso3: BRA

# Regenerate config
python config_builder.py config_MY_COUNTRY.yaml

# Re-upload to Databricks
databricks workspace import config.json /Workspace/Users/you/geospatial_pipeline/config.json --overwrite

# Re-run job (it will use new config)
databricks jobs run-now --job-id YOUR_JOB_ID
```

### To Add New Parameters

```bash
# Edit YAML
vim config_MY_COUNTRY.yaml
# Add new parameter under params:

params:
  new_parameter: value

# Regenerate
python config_builder.py config_MY_COUNTRY.yaml

# Update scripts if needed to use new parameter
```

---

## 📚 Next Steps

After successful setup:

1. **Read documentation:**
   - `PIPELINE_GUIDE.md` - Complete pipeline documentation
   - `CONFIG_GUIDE.md` - Configuration system details
   - `databricks_jobs/README.md` - Job deployment options

2. **Customize for your needs:**
   - Adjust performance parameters
   - Add custom validation
   - Integrate with your data sources

3. **Set up CI/CD** (optional):
   - Use Approach 3 (Asset Bundles)
   - Configure GitHub Actions
   - Automate deployments

4. **Scale to multiple countries:**
   - Create config per country
   - Run jobs in parallel
   - Aggregate results

---

## ✅ Checklist

Before running the pipeline, ensure:

- [ ] Cloned repository
- [ ] Created country-specific `config.yaml`
- [ ] Generated `config.json` successfully
- [ ] Uploaded all files to Databricks Workspace
- [ ] Created Databricks job
- [ ] Have cluster ID and it's running
- [ ] Have access to all input files (proportions, TSI, admin, tiles)
- [ ] Have write permissions to output volume
- [ ] Tested Task 0 generates config correctly

---

## 🆘 Getting Help

**If you're stuck:**

1. Check troubleshooting section above
2. Review `PIPELINE_GUIDE.md` for task-specific help
3. Verify your config with: `python config_builder.py config.yaml --summary`
4. Check Databricks job logs for error messages

**Common resources:**
- Pipeline docs: `PIPELINE_GUIDE.md`
- Config docs: `CONFIG_GUIDE.md`
- Job deployment: `databricks_jobs/README.md`

---

**🎉 Congratulations! You're ready to run the geospatial pipeline!**

The pipeline will:
1. ✅ Auto-generate config from your YAML (Task 0)
2. ✅ Load your proportions/TSI data (Task 1)
3. ✅ Generate 2km grids for your country (Task 2)
4. ✅ Download GHSL tiles (Task 3)
5. ✅ Extract building statistics (Task 4)
6. ✅ Calculate TSI estimates (Task 5)
7. ✅ Create analysis views (Task 6)
8. ✅ Export results to CSV + Excel (Task 7)

**All fully automated!** 🚀
