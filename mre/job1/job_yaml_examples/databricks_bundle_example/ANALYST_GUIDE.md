# Analyst Quick Start Guide

This guide helps analysts set up and run the Building Data Enrichment pipeline.

## 📋 Prerequisites

1. **Databricks CLI** installed:
   ```bash
   pip install databricks-cli
   ```

2. **Databricks access** configured:
   ```bash
   databricks configure --token
   ```

3. **Your input data files ready**:
   - Country proportions CSV (e.g., `IND_NOS_storey_mapping.csv`)
   - TSI multipliers CSV (`tsi.csv`)
   - Tile footprint shapefile (small, < 50 MB)

## 🚀 Setup Steps

### Step 1: Place Your Input Data

Navigate to the data directory:
```bash
cd databricks_bundle_example/data/inputs/
```

#### Add Your Files:

```bash
# 1. Proportions CSV (your country data)
cp /your/path/IND_NOS_storey_mapping.csv proportions/

# 2. TSI multipliers
cp /your/path/tsi.csv multipliers/

# 3. Tile footprint shapefile (if < 50 MB)
cp /your/path/GHSL2_0_MWD_L1_tile_schema_land.* reference_data/tile_footprint/
```

**Verify files are in place:**
```bash
ls -lh proportions/
ls -lh multipliers/
ls -lh reference_data/tile_footprint/
```

---

### Step 2: Configure for Your Country

Edit the main configuration file:

```bash
cd databricks_bundle_example/
vim config.yaml
```

**Update these key settings:**

```yaml
# Country to process
country:
  iso3: IND  # Change to your country (USA, BRA, etc.)

# Input file paths
inputs:
  # ✅ Files in bundle (small, < 50 MB)
  proportions_csv: ${workspace.root_path}/files/data/inputs/proportions/IND_NOS_storey_mapping.csv
  tsi_csv: ${workspace.root_path}/files/data/inputs/multipliers/tsi.csv
  tile_footprint: ${workspace.root_path}/files/data/inputs/reference_data/tile_footprint/GHSL2_0_MWD_L1_tile_schema_land.gpkg

  # ❌ Large files (keep in Volumes - world shapefile ~250 MB)
  admin_boundaries: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg
```

**Important notes:**
- `${workspace.root_path}` automatically points to the uploaded bundle files
- Large files (like world boundaries) should reference Databricks Volumes paths
- Match the filename exactly (including your country code)

---

### Step 3: Update Databricks Workspace Settings

Edit bundle configuration:

```bash
vim databricks.yml
```

**Update your email and workspace paths:**

```yaml
variables:
  workspace_path: /Workspace/Users/YOUR_EMAIL@company.com/your_project/scripts
  email_notifications: YOUR_EMAIL@company.com
```

---

### Step 4: Validate Bundle

Check everything is configured correctly:

```bash
databricks bundle validate
```

**Expected output:**
```
✅ Configuration is valid
✅ All paths resolved
✅ No errors found
```

**If you see errors:**
- Check file paths in `config.yaml`
- Verify all required files are present
- Ensure databricks.yml has correct workspace paths

---

### Step 5: Deploy to Databricks

Deploy to development environment:

```bash
databricks bundle deploy
```

**What happens:**
1. ✅ Bundle uploads to Databricks workspace
2. ✅ Your input files (CSVs, shapefiles) are uploaded
3. ✅ Job and cluster configs are created
4. ✅ Scripts are uploaded to workspace

**Expected output:**
```
Uploading databricks_bundle_example to /Workspace/Users/...
✓ Successfully uploaded
✓ Job created: building_enrichment_IND
```

---

### Step 6: Run the Pipeline

```bash
# Run the full pipeline
databricks bundle run building_enrichment_IND
```

**Monitor progress:**
1. Go to Databricks UI
2. Navigate to **Workflows**
3. Find your job: "Building Data Enrichment - IND [development]"
4. Click to see task progress

**Pipeline tasks:**
```
Task 0: Config Generation   (generates config.json from config.yaml)
    ↓
Task 1: Import Proportions  (loads your CSV to Delta)
    ↓
Task 2: Grid Generation     (creates grid cells)
    ↓
Task 3: Tile Download       (downloads GHSL data)
    ↓
Task 4: Raster Stats        (extracts building counts)
    ↓
Task 5: Post-Processing     (calculates estimates)
    ↓
Task 6: Create Views        (generates SQL views)
    ↓
Task 7: Export              (exports final results)
```

---

## 📊 Accessing Results

After successful run:

### Delta Tables

Access in Databricks SQL or notebook:

```sql
-- View final estimates
SELECT * FROM prp_mr_bdap_projects.geospatialsolutions.building_enrichment_output
LIMIT 10;

-- View grid centroids
SELECT * FROM prp_mr_bdap_projects.geospatialsolutions.grid_centroids
LIMIT 10;
```

### Exported Files

Check Volumes for exported CSVs and Excel files:
```
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/outputs/exports/
```

---

## 🔧 Common Scenarios

### Switching to a Different Country

```bash
# 1. Add new country's proportions CSV
cp /path/to/USA_NOS_storey_mapping.csv data/inputs/proportions/

# 2. Update config.yaml
vim config.yaml
# Change:
#   iso3: IND  →  iso3: USA
#   proportions_csv: IND_NOS... → USA_NOS...

# 3. Redeploy
databricks bundle deploy

# 4. Run
databricks bundle run building_enrichment_USA
```

### Updating TSI Multipliers

```bash
# 1. Replace TSI file
cp /path/to/new_tsi.csv data/inputs/multipliers/tsi.csv

# 2. Redeploy
databricks bundle deploy

# 3. Run (Task 0 will pick up new TSI values)
databricks bundle run building_enrichment_IND
```

### Testing with Sample Data

```bash
# 1. Create small test CSV (first 1000 rows)
head -n 1000 IND_NOS_storey_mapping.csv > data/inputs/proportions/IND_NOS_test.csv

# 2. Update config.yaml
vim config.yaml
# Change: proportions_csv: ...IND_NOS_test.csv

# 3. Deploy and run with test data
databricks bundle deploy
databricks bundle run building_enrichment_IND
```

---

## ⚠️ Important Notes

### File Size Limits

| File Type | Max Size for Bundle | If Larger |
|-----------|---------------------|-----------|
| **CSV files** | < 50 MB | ✅ Can bundle |
| **Shapefiles (tile footprint)** | < 50 MB | ✅ Can bundle |
| **World boundaries** | ANY (your file: 250 MB!) | ❌ Use Volumes only |

**Your world shapefile (RMS_Admin0_geozones.gpkg) at 250 MB:**
- ❌ **DO NOT add to bundle**
- ✅ **Keep in Volumes**:
  ```
  /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/
  ```

### Task 0: Automatic Config Generation

**You don't need to run `config_builder.py` manually!**

The pipeline automatically:
1. Runs Task 0 first
2. Generates `config.json` from `config.yaml`
3. All subsequent tasks use the fresh config

**This means:**
- ✅ Config always matches your YAML
- ✅ No stale config issues
- ✅ Easy for new analysts to use

---

## ❓ Troubleshooting

### "File not found" Error

**Problem**: Task fails with file not found.

**Solution**:
```bash
# Check files are in bundle
ls -lh data/inputs/proportions/
ls -lh data/inputs/multipliers/
ls -lh data/inputs/reference_data/tile_footprint/

# Verify paths in config.yaml match actual filenames
cat config.yaml | grep -A 5 "inputs:"
```

### "File too large" Git Error

**Problem**: Git refuses to commit large file.

**Solution**:
```bash
# Check file size
du -sh data/inputs/reference_data/tile_footprint/

# If > 50 MB:
# 1. Remove from bundle
git rm data/inputs/reference_data/tile_footprint/*.shp

# 2. Place in Databricks Volumes instead
# 3. Update config.yaml to reference Volume path
```

### World Shapefile Missing

**Problem**: Pipeline can't find `RMS_Admin0_geozones.gpkg`.

**Solution**:
```bash
# This file should be in Databricks Volumes, not bundle
# Upload to Volumes if not present:

# In Databricks UI or via CLI:
databricks fs cp /local/path/RMS_Admin0_geozones.gpkg \
  dbfs:/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/

# Verify path in config.yaml points to Volume:
admin_boundaries: /Volumes/.../admin/RMS_Admin0_geozones.gpkg
```

### Task 0 Config Generation Fails

**Problem**: Task 0 fails to generate config.

**Solution**:
```bash
# Check config.yaml syntax
cat config.yaml

# Validate YAML format
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check config_builder.py is in src/
ls -lh src/config_builder.py
```

---

## 📖 Additional Resources

**Documentation:**
- **README.md** - Bundle overview and deployment
- **STRUCTURE.md** - Detailed structure explanation
- **DATA_ORGANIZATION.md** - Where to place different data types
- **data/inputs/README.md** - Input data guidelines

**Need Help?**
- Check Databricks job logs for error details
- Review task-specific READMEs in `data/inputs/`
- Consult your team's data engineer

---

## ✅ Quick Checklist

Before running:

- [ ] Input files placed in `data/inputs/` directories
- [ ] `config.yaml` updated with your country and file paths
- [ ] `databricks.yml` updated with your email and workspace path
- [ ] Bundle validated: `databricks bundle validate`
- [ ] Bundle deployed: `databricks bundle deploy`
- [ ] World shapefile (250 MB) in Volumes, not bundle
- [ ] All file paths in config.yaml are correct

Ready to run:
```bash
databricks bundle run building_enrichment_IND
```

---

**Good luck with your analysis! 🚀**
