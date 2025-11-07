# Data Organization: Input Files Strategy

## Question: Where Should Input Files Go?

When deploying a Databricks bundle, you have different types of input data:

1. **Variable data** (changes frequently): proportions CSVs, TSI tables
2. **Reference data** (static): world shapefiles, tile footprints
3. **Generated data** (created by pipeline): Delta tables, outputs

This guide explains where each type should live.

---

## 📊 Input File Categories

### Category 1: Variable Data (Changes Per Country/Run)

**Examples from your pipeline:**
- `IND_NOS_storey_mapping_041125.csv` - India proportions
- `USA_NOS_storey_mapping.csv` - USA proportions (if you add it)
- `tsi.csv` - TSI multipliers (might change)

**Characteristics:**
- 🔄 Changes frequently (per country, per analysis)
- 📦 Relatively small (< 50 MB typically)
- 🎯 Specific to a pipeline run
- 👥 Used by one project/team

**Recommended Location: Databricks Volumes (Country-Specific Folder)**

```
/Volumes/catalog/schema/volume_name/
└── data/
    └── inputs/
        ├── proportions/
        │   ├── IND_NOS_storey_mapping_041125.csv
        │   ├── USA_NOS_storey_mapping.csv
        │   └── BRA_NOS_storey_mapping.csv
        └── multipliers/
            └── tsi.csv
```

**Why Volumes?**
- ✅ Easy to update (just upload new file)
- ✅ Accessible to all tasks in job
- ✅ Can be large files
- ✅ Version-controlled via file naming (IND_v2.csv)
- ✅ Shared across bundle deployments

**Alternative: Bundle `data/` folder (if files are small)**

```
databricks_bundle_example/
├── data/
│   ├── proportions/
│   │   └── IND_NOS_storey_mapping.csv   # Include if < 5 MB
│   └── multipliers/
│       └── tsi.csv
```

**When to bundle:**
- File is small (< 5 MB)
- Changes with code (tightly coupled)
- Want self-contained deployment

**When NOT to bundle:**
- File is large (> 5 MB)
- Changes independently of code
- Shared across multiple jobs
- Binary files (git doesn't handle well)

---

### Category 2: Reference Data (Static)

**Examples from your pipeline:**
- `RMS_Admin0_geozones.gpkg` - **World administrative boundaries**
- `GHSL2_0_MWD_L1_tile_schema_land.shp` - **Tile footprint**

**Characteristics:**
- 🔒 Rarely changes (months/years between updates)
- 🌍 Used across multiple countries/projects
- 📦 Can be large (100+ MB)
- 👥 Shared reference data

**Recommended Location: Unity Catalog Volumes (Shared Location)**

```
/Volumes/catalog/schema/reference_data/
└── inputs/
    ├── admin/
    │   └── RMS_Admin0_geozones.gpkg        # World shapefile
    └── tiles/
        └── GHSL2_0_MWD_L1_tile_schema_land.shp   # Tile footprint
```

**Why separate shared location?**
- ✅ **One source of truth** (don't duplicate 100 MB shapefile per project)
- ✅ **Easy to update** (update once, all jobs benefit)
- ✅ **Version control** (can keep old versions: `admin_v1/`, `admin_v2/`)
- ✅ **Governed access** (Unity Catalog permissions)

**DO NOT bundle these files:**
- ❌ Too large for git
- ❌ Binary files cause git bloat
- ❌ Rarely change
- ❌ Should be shared across projects

---

### Category 3: Generated/Output Data

**Examples:**
- Delta tables (proportions, grid_centroids, estimates)
- Exports (CSV, Excel)
- Intermediate results

**Recommended Location: Unity Catalog Tables & Volumes**

```
Unity Catalog Tables:
  catalog.schema.proportions_IND
  catalog.schema.grid_centroids_IND
  catalog.schema.estimates_combined_IND

Unity Catalog Volumes (exports):
  /Volumes/catalog/schema/outputs/
  └── exports/
      ├── IND_estimates_2024-11-07.csv
      └── IND_estimates_2024-11-07.xlsx
```

---

## 🏗️ Recommended Directory Structure

### Option A: Hybrid (Recommended)

**Best for most cases** - Small variable data in bundle, large/reference in Volumes

```
Your Setup:
├── Databricks Bundle (Git Repository)
│   ├── databricks.yml
│   ├── config.yaml
│   ├── resources/
│   ├── src/
│   └── data/                          # Optional: Small variable data only
│       └── multipliers/
│           └── tsi.csv                # < 5 MB, changes with code
│
└── Databricks Volumes (Centralized Storage)
    ├── /Volumes/.../inputs/
    │   ├── proportions/               # Variable data (per country)
    │   │   ├── IND_NOS_storey_mapping.csv
    │   │   └── USA_NOS_storey_mapping.csv
    │   ├── admin/                     # Reference data (shared)
    │   │   └── RMS_Admin0_geozones.gpkg
    │   └── tiles/                     # Reference data (shared)
    │       └── GHSL2_0_MWD_L1_tile_schema_land.shp
    └── /Volumes/.../outputs/          # Generated data
        └── exports/
```

**In your config.yaml:**
```yaml
inputs:
  # Variable data - in Volumes (easy to update per country)
  proportions_csv: /Volumes/catalog/schema/external/jrc/data/inputs/proportions/IND_NOS_storey_mapping.csv

  # Small variable data - could be in bundle
  tsi_csv: /Volumes/catalog/schema/external/jrc/data/inputs/multipliers/tsi.csv
  # Or: ${workspace.root_path}/files/data/multipliers/tsi.csv  (if bundled)

  # Reference data - in shared Volumes location
  admin_boundaries: /Volumes/catalog/schema/reference_data/admin/RMS_Admin0_geozones.gpkg
  tile_footprint: /Volumes/catalog/schema/reference_data/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp
```

---

### Option B: Everything in Volumes (Current Approach)

**Best for** - Large files, frequently changing data

```
All data in Volumes:
/Volumes/prp_mr_bdap_projects/geospatialsolutions/
├── external/jrc/data/
│   ├── inputs/
│   │   ├── proportions/               # Variable data
│   │   │   └── IND_NOS_storey_mapping.csv
│   │   ├── multipliers/               # Variable data
│   │   │   └── tsi.csv
│   │   ├── admin/                     # Reference data
│   │   │   └── RMS_Admin0_geozones.gpkg
│   │   └── tiles/                     # Reference data
│   │       ├── GHSL2_0_MWD_L1_tile_schema_land.shp
│   │       ├── built_c/               # Downloaded tiles
│   │       └── smod/                  # Downloaded tiles
│   └── outputs/
│       └── exports/
```

**Pros:**
- ✅ Centralized (everything in one place)
- ✅ Large files supported
- ✅ Easy to manage in Databricks UI
- ✅ Your current working approach

**Cons:**
- ⚠️ Not self-contained (bundle needs Volume setup)
- ⚠️ Manual setup required before deployment

---

### Option C: Everything in Bundle (Not Recommended for You)

**Only for** - Very small datasets, demo projects

```
databricks_bundle_example/
├── databricks.yml
├── src/
└── data/                              # All data in bundle
    ├── proportions/
    │   └── IND_NOS_storey_mapping.csv
    ├── multipliers/
    │   └── tsi.csv
    ├── admin/
    │   └── RMS_Admin0_geozones.gpkg   # ❌ Too large for git!
    └── tiles/
        └── GHSL_tile_schema.shp       # ❌ Too large for git!
```

**Why NOT recommended for you:**
- ❌ Shapefiles are large (100+ MB)
- ❌ Git bloat with binary files
- ❌ Hard to update data independently
- ❌ Bundle size becomes huge

---

## 🎯 Your Specific Case: Recommendations

Based on your pipeline:

### 1. **Proportions CSV** (IND_NOS_storey_mapping.csv)
**Location:** Volumes - Country-specific folder

```bash
# Current location is fine:
/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/data/IND_NOS_storey_mapping_041125.csv

# Better: Move to Volumes for easier sharing
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/proportions/IND_NOS_storey_mapping.csv
```

**Why:**
- Changes per country (IND, USA, BRA, etc.)
- Need different versions for different runs
- Easy to update without redeploying bundle

**Workflow:**
```bash
# Add new country:
1. Upload USA_NOS_storey_mapping.csv to Volumes
2. Update config.yaml: iso3: USA
3. Update config.yaml: proportions_csv path to USA file
4. Redeploy bundle
5. Task 0 generates config with USA settings
```

---

### 2. **TSI CSV** (tsi.csv)
**Location:** Volumes (current) OR bundle `data/` (if small & stable)

**Current (Volumes):**
```yaml
tsi_csv: /Volumes/.../data/inputs/multipliers/tsi.csv
```

**Alternative (Bundle - if file is small):**
```yaml
# If tsi.csv is < 5 MB and changes rarely:
tsi_csv: ${workspace.root_path}/files/data/multipliers/tsi.csv
```

**Recommendation:** Keep in Volumes for now (easier to update)

---

### 3. **Admin Boundaries** (RMS_Admin0_geozones.gpkg) - WORLD SHAPEFILE
**Location:** ✅ **Volumes - Shared Reference Location**

```bash
# Current location (OK):
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg

# Better: Move to reference_data volume (if you create one)
/Volumes/prp_mr_bdap_projects/reference_data/admin/world_boundaries_v1.gpkg
```

**Why:**
- ✅ **Static** (doesn't change per country)
- ✅ **Shared** across all countries (IND, USA, BRA all use same world shapefile)
- ✅ **Large file** (probably 50-200 MB)
- ✅ **Reference data** (update once globally)

**Recommendation:**
- Keep in Volumes (current approach is good)
- ❌ **DO NOT include in bundle** (too large for git)
- Consider creating a `reference_data` volume separate from job-specific data

---

### 4. **Tile Footprint** (GHSL2_0_MWD_L1_tile_schema_land.shp)
**Location:** ✅ **Volumes - Shared Reference Location**

```bash
# Current location (OK):
/Volumes/.../data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp

# Same as admin boundaries - this is reference data
```

**Why:**
- ✅ **Static** (GHSL tile schema doesn't change per country)
- ✅ **Shared** (all countries use same tile schema)
- ✅ **Binary shapefile** (not git-friendly)

**Recommendation:**
- Keep in Volumes
- ❌ **DO NOT include in bundle**

---

## 📋 Decision Matrix

| File Type | Size | Changes? | Shared? | **Recommended Location** |
|-----------|------|----------|---------|-------------------------|
| Proportions CSV | < 50 MB | Per country | No | **Volumes** (country-specific) |
| TSI CSV | < 5 MB | Rarely | Maybe | **Volumes** (or bundle if tiny) |
| World Shapefile | 50-200 MB | Rarely | **Yes** | **Volumes** (reference_data) |
| Tile Footprint | 10-50 MB | Never | **Yes** | **Volumes** (reference_data) |

---

## 🔧 Implementation: Adding `data/` to Bundle (Optional)

If you want to include small files in the bundle:

### 1. Create `data/` directory structure

```bash
cd databricks_bundle_example/

mkdir -p data/multipliers
mkdir -p data/proportions

# Copy small files (< 5 MB only)
cp /path/to/tsi.csv data/multipliers/
```

### 2. Update `.gitignore`

```gitignore
# .gitignore

# Generated files
config.json
*.json.backup

# Large data files (DO NOT COMMIT)
data/**/*.gpkg        # Shapefiles
data/**/*.shp         # Shapefiles
data/**/*.tif         # Rasters
data/**/*.tiff        # Rasters

# Allow small CSVs (< 5 MB)
!data/**/*.csv        # CSV files OK
```

### 3. Update `databricks.yml` to sync data

```yaml
# databricks.yml

sync:
  include:
    - "src/**/*.py"
    - "config.yaml"
    - "data/**/*.csv"     # Include small CSVs
  exclude:
    - "data/**/*.gpkg"    # Exclude large shapefiles
    - "data/**/*.shp"     # Exclude shapefiles
    - "config.json"
```

### 4. Reference in config.yaml

```yaml
# config.yaml

inputs:
  # Small bundled file
  tsi_csv: ${workspace.root_path}/files/data/multipliers/tsi.csv

  # Large files in Volumes
  proportions_csv: /Volumes/.../inputs/proportions/IND_NOS_storey_mapping.csv
  admin_boundaries: /Volumes/.../reference_data/admin/world_boundaries.gpkg
  tile_footprint: /Volumes/.../reference_data/tiles/GHSL_tile_schema.shp
```

---

## ✅ Final Recommendation for Your Pipeline

### Keep Current Approach (Volumes) with Small Refinement

```
Databricks Bundle (Git):
├── src/                              ✅ All Python scripts
├── config.yaml                       ✅ Configuration (references Volume paths)
├── databricks.yml                    ✅ Bundle config
└── resources/                        ✅ Job/cluster YAML

Databricks Volumes:
├── /Volumes/.../inputs/
│   ├── proportions/                  ✅ Variable data (per country)
│   │   ├── IND_NOS_storey_mapping.csv
│   │   └── USA_NOS_storey_mapping.csv
│   ├── multipliers/                  ✅ Variable data
│   │   └── tsi.csv
│   └── reference_data/               ✅ NEW: Shared reference data
│       ├── admin/
│       │   └── world_boundaries.gpkg  # Move here
│       └── tiles/
│           └── GHSL_tile_schema.shp   # Move here
└── /Volumes/.../outputs/             ✅ Generated data
```

### Changes to Make:

1. **Create reference_data folder in Volumes** (optional but cleaner):
   ```bash
   # In Databricks:
   /Volumes/prp_mr_bdap_projects/reference_data/
   ├── admin/
   │   └── RMS_Admin0_geozones.gpkg
   └── tiles/
       └── GHSL2_0_MWD_L1_tile_schema_land.shp
   ```

2. **Update config.yaml** to reference new locations:
   ```yaml
   inputs:
     proportions_csv: /Volumes/.../inputs/proportions/${iso3}_NOS_storey_mapping.csv
     tsi_csv: /Volumes/.../inputs/multipliers/tsi.csv
     admin_boundaries: /Volumes/.../reference_data/admin/RMS_Admin0_geozones.gpkg
     tile_footprint: /Volumes/.../reference_data/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp
   ```

3. **Keep files in Volumes** (don't add to bundle)

---

## 🎓 Summary

**Your Question:** Where should input files go? Should they be in bundle?

**Answer:**

| File | In Bundle? | Location | Why |
|------|-----------|----------|-----|
| **Proportions CSV** | ❌ No | Volumes (country-specific) | Changes per country, easy to update |
| **TSI CSV** | ❌ No | Volumes | Could bundle if tiny, but Volumes easier |
| **World Shapefile** | ❌ **NEVER** | Volumes (reference_data) | Large, static, shared - perfect for Volumes |
| **Tile Footprint** | ❌ **NEVER** | Volumes (reference_data) | Large, static, shared - perfect for Volumes |

**Key Insight:**
- ✅ **Static, shared reference data** (world shapefile) → Volumes, shared location
- ✅ **Variable data** (proportions per country) → Volumes, easy to update
- ❌ **Large files** → NEVER in git/bundle
- ✅ **Code & configs** → Bundle (git)

**Your current approach is correct!** Keep data in Volumes, code in bundle.
