# Geospatial Solutions Pipeline - Complete Guide

**Complete documentation from configuration to final export**

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration System](#configuration-system)
4. [Pipeline Tasks](#pipeline-tasks)
5. [Complete Workflow](#complete-workflow)
6. [Data Flow](#data-flow)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Overview

### What This Pipeline Does

This pipeline processes **Global Human Settlement Layer (GHSL)** satellite data to generate building density estimates and Total Sum Insured (TSI) calculations at a 2km grid resolution for any country.

**Input:** Raw GHSL raster data + building proportions + TSI multipliers
**Output:** Grid-level building estimates by type (RES/COM/IND) and storey levels with TSI calculations

### Key Features

- ✅ **ISO3-aware**: Process any country by changing one config value
- ✅ **2km grid resolution**: Stable, reproducible cell IDs
- ✅ **Building classification**: RES (11-15) and COM (21-25)
- ✅ **Storey distribution**: 7 bins (1, 2, 3, 4-5, 6-8, 9-20, 20+)
- ✅ **Urban classification**: SMOD data integration (Rural/Urban/Suburban)
- ✅ **TSI calculations**: Floor space and insurance estimates
- ✅ **Automated**: Downloads tiles, processes data, exports results
- ✅ **Scalable**: Parallel processing with configurable workers

### Architecture

```
YAML Config (20 lines)
        ↓
   config_builder.py
        ↓
Full Config (50+ values)
        ↓
7 Sequential Tasks
        ↓
Final Exports (CSV + Excel)
```

---

## Quick Start

### 1. Configure for Your Country

Edit `config.yaml`:

```yaml
country:
  iso3: USA  # Change to your country

inputs:
  proportions_csv: /path/to/USA_proportions.csv
  tsi_csv: /path/to/USA_tsi.csv
  # Update other paths...
```

### 2. Generate Configuration

```bash
cd mre/job1
python config_builder.py config.yaml
```

Output:
```
✅ Generated configuration written to: config.json
✅ Configurations are identical!
```

### 3. Run Tasks Sequentially

```bash
# Task 1: Load multipliers
python task1_proportions_to_delta.py --config_path config.json

# Task 2: Generate grid
python task2_grid_generation.py --config_path config.json

# Task 3: Download tiles
python task3_tile_downloader.py --config_path config.json

# Task 4: Extract raster stats
python task4_raster_stats.py --config_path config.json

# Task 5: Post-processing & TSI
python task5_post_processing.py --config_path config.json

# Task 6: Create views
python task6_create_views.py --config_path config.json

# Task 7: Export results
python task7_export.py --config_path config.json
```

### 4. Find Your Results

**Delta Tables:**
- `{catalog}.{schema}.building_enrichment_output_{iso3}`

**CSV Exports:**
- `{output_dir}/exports/FULL_{ISO3}/building_enrichment_output_{ISO3}_FULL.csv`

**Excel Summary:**
- `{output_dir}/exports/building_summary_country_layout_{ISO3}.xlsx`

---

## Configuration System

### Understanding the YAML Configuration

The pipeline uses a **simplified YAML configuration** that auto-generates all derived paths and table names.

#### config.yaml Structure

```yaml
# ============================================================================
# PROJECT SETTINGS (Databricks environment)
# ============================================================================
project:
  catalog: prp_mr_bdap_projects              # Databricks catalog
  schema: geospatialsolutions                # Databricks schema
  volume_root: /Volumes/.../data             # Base volume path

# ============================================================================
# COUNTRY SETTINGS (main variable to change)
# ============================================================================
country:
  iso3: IND  # ISO 3166-1 alpha-3 code (USA, BRA, GBR, etc.)

# ============================================================================
# INPUT FILES (paths to your input data)
# ============================================================================
inputs:
  proportions_csv: /path/to/proportions.csv
  tsi_csv: /path/to/tsi.csv
  admin_boundaries: /path/to/boundaries.gpkg
  tile_footprint: /path/to/tile_schema.shp

# ============================================================================
# PROCESSING PARAMETERS (tune for performance)
# ============================================================================
params:
  cell_size: 2000                   # Grid cell size (meters)
  datasets: built_c,smod            # GHSL datasets
  chunk_size: 10000                 # Processing chunk size
  max_workers: 8                    # Parallel workers
  # ... more parameters ...
```

### What Gets Auto-Generated

From the YAML above, `config_builder.py` generates **50+ configuration values**:

#### 📁 **Folder Paths** (all derived from `volume_root`)

```
/Volumes/.../data/
├── inputs/
│   ├── tiles/
│   │   ├── built_c/          ← Auto-generated
│   │   └── smod/             ← Auto-generated
│   ├── admin/                ← Auto-generated
│   └── multipliers/          ← Auto-generated
└── outputs/                  ← Auto-generated
```

#### 📊 **Table Names** (all with consistent naming)

```
{catalog}.{schema}.building_enrichment_proportions_input
{catalog}.{schema}.building_enrichment_tsi_input
{catalog}.{schema}.grid_centroids
{catalog}.{schema}.download_status
{catalog}.{schema}.grid_counts
{catalog}.{schema}.building_enrichment_output
```

#### ⚙️ **Derived Settings**

- `admin_field`: "ISO3"
- `admin_value`: {iso3}
- `tile_id_field`: "tile_id"
- All CRS settings, paths, flags, etc.

### Using config_builder.py

#### Basic Generation

```bash
python config_builder.py config.yaml
```

Creates `config.json` with all values.

#### View Summary Without Writing

```bash
python config_builder.py config.yaml --summary
```

Output:
```
================================================================================
CONFIGURATION SUMMARY
================================================================================

📊 Project:
   Catalog: prp_mr_bdap_projects
   Schema: geospatialsolutions
   Country: IND

📋 Delta Tables Generated: 7 tables
🗂️  Folder Structure: 5 folders
...
```

#### Validate Against Existing Config

```bash
python config_builder.py config.yaml --validate
```

Shows differences between generated and existing configs.

#### Custom Output Path

```bash
python config_builder.py config.yaml --output config_usa.json
```

### Input File Requirements

#### 1. Proportions CSV

**Purpose:** Building storey distribution by building type

**Format:**
```csv
built,1,2,3,4_5,6_8,9_20,20
11,0.30,0.25,0.20,0.15,0.05,0.03,0.02
12,0.25,0.30,0.20,0.15,0.05,0.03,0.02
...
```

**Requirements:**
- Column `built`: Building type codes (11-15 for RES, 21-25 for COM)
- Storey columns: 1, 2, 3, 4_5, 6_8, 9_20, 20
- Values must sum to 1.0 (or 100% if percentages)
- Supports comma decimals (0,49) or period decimals (0.49)

#### 2. TSI CSV

**Purpose:** Total Sum Insured per square meter by building type

**Format:**
```csv
built,tsi_m2
11,500.00
12,550.00
13,600.00
...
```

**Requirements:**
- Column `built`: Building type codes
- Column `tsi_m2`: TSI value per square meter

#### 3. Admin Boundaries (GeoPackage)

**Purpose:** Country boundaries for grid generation and masking

**Requirements:**
- Format: GeoPackage (.gpkg)
- Must have ISO3 field with country codes
- Valid geometries

#### 4. Tile Footprint (Shapefile)

**Purpose:** GHSL tile coverage for determining which tiles to download

**Requirements:**
- Format: Shapefile (.shp)
- Must have `tile_id` field
- Coverage: Global GHSL tile schema

---

## Pipeline Tasks

### Task 1: Load Multipliers

**Script:** `task1_proportions_to_delta.py`

**Purpose:** Load proportions and TSI CSV files into Delta tables

**Input:**
- `proportions_csv_path`: Path to proportions CSV
- `tsi_csv_path`: Path to TSI CSV

**Output:**
- Delta table: `{catalog}.{schema}.building_enrichment_proportions_input`
- Delta table: `{catalog}.{schema}.building_enrichment_tsi_input`

**Features:**
- Smart format detection (comma vs period decimals)
- Auto-normalization (percentages → decimals)
- Column mapping (handles variations like "4-5" vs "4_5")
- Validation (row sums must equal 1.0)

**Usage:**
```bash
python task1_proportions_to_delta.py --config_path config.json
```

**What It Does:**

1. Reads proportions CSV
2. Detects numeric format (comma/period decimals, percentages)
3. Normalizes to 0.0-1.0 range
4. Maps column headers to canonical names
5. Validates row sums
6. Writes to Delta table

**Expected Output:**
```
================================================================================
TASK 1: Load Multipliers to Delta
================================================================================
Catalog: prp_mr_bdap_projects
Schema: geospatialsolutions
ISO3: IND

[1/2] Processing Proportions CSV
  ✓ Loaded 15 rows
  ✓ Detected decimal format
  ✓ Validated row sums
  ✓ Wrote to Delta table

[2/2] Processing TSI CSV
  ✓ Loaded 15 rows
  ✓ Wrote to Delta table

✅ Task 1 Complete
```

---

### Task 2: Generate Grid

**Script:** `task2_grid_generation.py`

**Purpose:** Generate 2km grid centroids covering country boundaries

**Input:**
- `admin_path`: Admin boundaries
- `tile_footprint_path`: GHSL tile footprint
- `cell_size`: Grid resolution (default: 2000m)

**Output:**
- CSV: `{volume_root}/outputs/grid_centroids.csv`
- Delta table: `{catalog}.{schema}.grid_centroids`

**Features:**
- Stable grid IDs based on Mollweide coordinates
- Only generates grids intersecting country boundaries
- Assigns GHSL tile_id to each grid cell
- Uses Mollweide projection for accurate equal-area grids

**Usage:**
```bash
python task2_grid_generation.py --config_path config.json
```

**What It Does:**

1. Loads admin boundaries for country (ISO3 filter)
2. Loads GHSL tile footprint
3. Finds tiles intersecting country
4. Generates 2km grid centroids in Mollweide projection
5. Assigns grid_id (stable, reproducible)
6. Assigns tile_id to each centroid
7. Exports to WGS84 (lat/lon) for output

**Grid ID Format:**
```
grid_id = f"X{x_moll}Y{y_moll}"
```
Example: `X-5438000Y3246000`

**Expected Output:**
```
================================================================================
TASK 2: Grid Generation
================================================================================
ISO3: IND
Cell Size: 2000m
Target CRS: ESRI:54009 (Mollweide)
Export CRS: EPSG:4326 (WGS84)

Loading data...
  ✓ Admin boundaries: 1 country
  ✓ Tile footprint: 12 tiles intersect

Generating grids...
  ✓ Generated 45,234 grid centroids
  ✓ Assigned tile IDs

Writing outputs...
  ✓ CSV: .../grid_centroids.csv
  ✓ Delta: grid_centroids

✅ Task 2 Complete
```

---

### Task 3: Download Tiles

**Script:** `task3_tile_downloader.py`

**Purpose:** Download GHSL raster tiles from JRC repository

**Input:**
- `grid_source`: Delta table with grid centroids (contains tile_id)
- `tiles_dest_root`: Destination for downloaded tiles
- `datasets`: Comma-separated datasets (e.g., "built_c,smod")

**Output:**
- Tiles: `{tiles_dest_root}/built_c/{tile_id}/*.tif`
- Tiles: `{tiles_dest_root}/smod/{tile_id}/*.tif`
- Delta table: `{catalog}.{schema}.download_status`

**Features:**
- Concurrent downloads (configurable workers)
- Automatic retries on failure
- Extracts zip files to TIFF
- Tracks download status in Delta table

**GHSL Datasets:**

**built_c** - Building Construction Layer
- **URL:** https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_C_GLOBE_R2023A/
- **Resolution:** 10m
- **Values:** Building type codes (11-15 RES, 21-25 COM)

**smod** - Settlement Model Layer
- **URL:** https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/
- **Resolution:** 1km
- **Values:** 0=Rural, 1=Urban, 2=Suburban

**Usage:**
```bash
python task3_tile_downloader.py --config_path config.json
```

**What It Does:**

1. Reads distinct tile_ids from grid_centroids table
2. For each tile and dataset:
   - Constructs download URL
   - Downloads zip file
   - Extracts TIFF files
   - Verifies extracted files exist
3. Writes download status to Delta table

**Expected Output:**
```
================================================================================
TASK 3: Tile Downloader
================================================================================
Datasets: built_c, smod
Tiles to download: 12
Concurrency: 3 workers
Retries: 2

Downloading tiles...
[1/12] R10_C15 (built_c)... ✓ 245 MB
[2/12] R10_C15 (smod)... ✓ 89 MB
[3/12] R10_C16 (built_c)... ✓ 238 MB
...

Download Summary:
  ✓ Successful: 24/24 (100%)
  ✗ Failed: 0/24 (0%)

✅ Task 3 Complete
```

---

### Task 4: Extract Raster Statistics

**Script:** `task4_raster_stats.py`

**Purpose:** Extract building counts from raster tiles at grid centroids

**Input:**
- `grid_source`: Grid centroids table
- `built_root`: Path to built_c tiles
- `smod_root`: Path to SMOD tiles (optional)
- `admin_path`: Admin boundaries for masking

**Output:**
- Delta table: `{catalog}.{schema}.grid_counts`

**Schema:**
```
grid_id    | built | count | urban | lat    | lon    | tile_id
-----------|-------|-------|-------|--------|--------|--------
X123Y456   | 11    | 450   | 1     | 28.5   | 77.2   | R10_C15
X123Y456   | 12    | 230   | 1     | 28.5   | 77.2   | R10_C15
...
```

**Features:**
- Tile-level parallelism (processes multiple tiles concurrently)
- Chunk-level parallelism (8 workers per tile)
- Local staging (copies tiles to local disk for faster reads)
- Boundary masking (skips points outside country)
- Memory-efficient windowed reads

**Building Classification:**

| Built Code | Category | Description |
|------------|----------|-------------|
| 11 | RES | Residential - Single family |
| 12 | RES | Residential - Multi-family |
| 13 | RES | Residential - Mixed use |
| 14 | RES | Residential - Temporary |
| 15 | RES | Residential - Other |
| 21 | COM | Commercial/Industrial - Offices |
| 22 | COM | Commercial/Industrial - Retail |
| 23 | COM | Commercial/Industrial - Industrial |
| 24 | COM | Commercial/Industrial - Transport |
| 25 | COM | Commercial/Industrial - Other |

**SMOD Urban Classification:**

| Value | Category | Description |
|-------|----------|-------------|
| 0 | Rural | Low density, scattered settlements |
| 1 | Urban | High density urban centers |
| 2 | Suburban | Medium density suburban areas |

**Usage:**
```bash
python task4_raster_stats.py --config_path config.json
```

**What It Does:**

1. Loads grid centroids grouped by tile_id
2. For each tile (parallel):
   - Stages tiles to local disk (optional)
   - Opens built_c and SMOD rasters
   - Processes grid points in chunks
   - For each point:
     - Checks if inside country boundary
     - Reads raster value at point location
     - Reads SMOD urban classification
     - Aggregates counts by (grid_id, built, urban)
3. Writes results to Delta table

**Optimizations:**

- **Tile parallelism:** 4 tiles processed concurrently (configurable)
- **Chunk parallelism:** 8 workers per tile for chunk processing
- **Chunk size:** 10,000 points per chunk (configurable)
- **Local staging:** Copies tiles to `/local_disk0` for faster I/O
- **Windowed reads:** Only reads small windows around each point

**Expected Output:**
```
================================================================================
TASK 4: Raster Statistics Extraction
================================================================================
Grid points: 45,234
Tiles: 12
Use SMOD: True
Chunk size: 10,000
Max workers: 8
Tile parallelism: 4

Processing tiles...
[1/12] R10_C15: 3,842 points
  Chunk 1/1... ✓ 3,842 points, 2.3s
  Extracted: 15,368 records

[2/12] R10_C16: 4,156 points
  Chunk 1/1... ✓ 4,156 points, 2.5s
  Extracted: 16,624 records
...

Summary:
  Total points: 45,234
  Inside boundary: 42,891 (94.8%)
  Outside boundary: 2,343 (5.2%)
  Total records: 171,564

Writing to Delta...
  ✓ Written: grid_counts

✅ Task 4 Complete
Time: 4m 32s
```

---

### Task 5: Post-Processing & TSI

**Script:** `task5_post_processing.py`

**Purpose:** Process grid counts to generate building estimates with TSI calculations

**Input:**
- `grid_count_table`: Grid counts from Task 4
- `proportions_table`: Proportions from Task 1
- `tsi_table`: TSI multipliers from Task 1

**Output:**
- Delta table: `{catalog}.{schema}.building_enrichment_output`

**Processing Steps:**

#### 1. **Pivot Building Counts**
```
From:
  grid_id | built | count | urban
  --------|-------|-------|------
  X123    | 11    | 450   | 1
  X123    | 12    | 230   | 1

To:
  grid_id | nr11_res | nr12_res | nr21_com | urban
  --------|----------|----------|----------|------
  X123    | 450      | 230      | 0        | 1
```

#### 2. **Filter Zero Buildings**
Removes rows where ALL building counts are zero

#### 3. **Add ID & Order**
- `ID`: Unique sequential ID
- `order_id`: Spatial ordering (sort by lat, lon)

#### 4. **Storey Distribution**
Apply proportions to distribute buildings by storey level:

```
nr11_res = 450 buildings
Proportions for built=11: {1: 0.30, 2: 0.25, 3: 0.20, ...}

Result:
  res_storey1 = 450 × 0.30 = 135
  res_storey2 = 450 × 0.25 = 112.5
  res_storey3 = 450 × 0.20 = 90
  ...
```

#### 5. **TSI Calculations**
Multiply storey counts by TSI rates:

```
res_storey1 = 135 buildings
TSI rate for built=11 = 500 $/m²

Result:
  res_storey1_TSI = 135 × 500 = 67,500
```

#### 6. **TSI Percentages**
Calculate percentage of total TSI by storey:

```
Total RES TSI = 150,000
res_storey1_TSI = 67,500

Result:
  res_storey1_tsi_perc = (67,500 / 150,000) × 100 = 45%
```

#### 7. **Imputation**
Fill missing TSI percentages with urban-specific averages

#### 8. **Column Organization**
Group related columns with block sums for validation

**Output Schema:**

```
Core Columns:
  - grid_id: Grid identifier
  - urban: Urban classification (0=Rural, 1=Urban, 2=Suburban)
  - lat: Latitude (WGS84)
  - lon: Longitude (WGS84)

Building Counts:
  - nr{built}_{lob}: e.g., nr11_res, nr12_res, nr21_com

Storey Distribution:
  - {lob}_storey{level}: e.g., res_storey1, res_storey2, com_storey4_5

TSI Values:
  - {lob}_storey{level}_TSI: TSI amount per storey

TSI Percentages:
  - {lob}_storey{level}_tsi_perc: Percentage of total TSI
```

**Usage:**
```bash
python task5_post_processing.py --config_path config.json
```

**Expected Output:**
```
================================================================================
TASK 5: Post-Processing & TSI Calculations
================================================================================

Step 1: Pivot building counts... ✓ 42,891 grids
Step 2: Filter zero buildings... ✓ Removed 1,234 grids (2.9%)
Step 3: Add ID and order... ✓
Step 4: Storey distribution... ✓ Applied proportions
Step 5: TSI calculations... ✓ Computed TSI values
Step 6: TSI percentages... ✓ Computed percentages
Step 7: Imputation... ✓ Filled 234 missing values
Step 8: Column organization... ✓

Final output:
  Rows: 41,657
  Columns: 127

Writing to Delta...
  ✓ Written: building_enrichment_output

✅ Task 5 Complete
```

---

### Task 6: Create Views

**Script:** `task6_create_views.py`

**Purpose:** Create per-LOB views with standardized column mapping

**Input:**
- `output_table`: Building enrichment output from Task 5

**Output:**
- View: `{catalog}.{schema}.building_enrichment_tsi_proportions_{iso3}_res_view`
- View: `{catalog}.{schema}.building_enrichment_tsi_proportions_{iso3}_com_view`
- View: `{catalog}.{schema}.building_enrichment_tsi_proportions_{iso3}_ind_view`

**View Schema:**

```sql
ID          -- Grid identifier
POINT_X     -- Longitude
POINT_Y     -- Latitude
ORDER_ID_XY -- Spatial order
1           -- Storey level 1 (TSI percentage)
2           -- Storey level 2
3           -- Storey level 3
5           -- Storey level 4-5 (mapped to representative 5)
7           -- Storey level 6-8 (mapped to representative 7)
10          -- Storey level 9-20 (mapped to representative 10)
40          -- Storey level 20+ (mapped to representative 40)
SUM         -- Total (sum of all storey percentages)
```

**Storey Mapping:**

| Original | Mapped | Representative |
|----------|--------|----------------|
| 1 | 1 | 1-storey |
| 2 | 2 | 2-storey |
| 3 | 3 | 3-storey |
| 4_5 | 5 | 5-storey (mid-point) |
| 6_8 | 7 | 7-storey (mid-point) |
| 9_20 | 10 | 10-storey (lower-mid) |
| 20 | 40 | 40-storey (high-rise) |

**CSV Compatibility:**

Uses `CAST(NULL AS <type>)` instead of plain `NULL` to prevent VOID type columns that cannot be exported to CSV.

**Usage:**
```bash
python task6_create_views.py --config_path config.json
```

**Expected Output:**
```
================================================================================
TASK 6: Create CSV-Compatible TSI Proportion Views
================================================================================
ISO3: ind
Input table: building_enrichment_output_ind

Creating views for LOBs: ['RES', 'COM', 'IND']

Creating CSV-compatible view: building_enrichment_tsi_proportions_ind_res_view
================================================================================
CREATE OR REPLACE VIEW building_enrichment_tsi_proportions_ind_res_view AS
SELECT
  `grid_id` AS ID,
  `lon` AS POINT_X,
  `lat` AS POINT_Y,
  `order_id` AS ORDER_ID_XY,
  COALESCE(`res_storey1_tsi_perc`, 0.0) AS `1`,
  COALESCE(`res_storey2_tsi_perc`, 0.0) AS `2`,
  ...
================================================================================
✅ Created view: building_enrichment_tsi_proportions_ind_res_view
✅ View is CSV-compatible (no VOID columns)

[Repeats for COM and IND]

================================================================================
VIEW CREATION COMPLETE
================================================================================
✅ All 3 views created successfully
✅ Views are CSV-compatible and ready for export
```

---

### Task 7: Export

**Script:** `task7_export.py`

**Purpose:** Export full datasets as CSV and generate Excel summary

**Input:**
- `output_table`: Building enrichment output
- Views from Task 6

**Output:**

**CSV Exports** (`{output_dir}/exports/FULL_{ISO3}/`):
- `building_enrichment_output_{ISO3}_FULL.csv` - Main output
- `building_enrichment_tsi_proportions_{ISO3}_RES_FULL.csv` - RES view
- `building_enrichment_tsi_proportions_{ISO3}_COM_FULL.csv` - COM view
- `building_enrichment_tsi_proportions_{ISO3}_IND_FULL.csv` - IND view

**Excel Summary** (`{output_dir}/exports/`):
- `building_summary_country_layout_{ISO3}.xlsx`

**Excel Sheets:**

1. **Country Summary** - Aggregated totals by LOB and storey
2. **Urban Breakdown** - By urban classification (Rural/Urban/Suburban)
3. **Grid-Level Data** - Sample of detailed grid data

**Features:**
- Overwrites previous exports (keeps only latest)
- Handles large datasets (Spark → Pandas fallback)
- Cleans incompatible column types
- Formats Excel with styling and formulas

**Usage:**
```bash
python task7_export.py --config_path config.json
```

**Expected Output:**
```
================================================================================
BUILDING ENRICHMENT FULL DATASET EXPORT + EXCEL SUMMARY
================================================================================
ISO3:          IND
Catalog:       prp_mr_bdap_projects
Schema:        geospatialsolutions
Export folder: .../exports/FULL_IND
================================================================================

🧹 Cleaning old exports from .../exports/FULL_IND
  ✓ Cleaned

================================================================================
EXPORTING FULL DATASETS
================================================================================

[1/4] Export main output
  Table: building_enrichment_output_ind
  Checking 127 columns for CSV compatibility...
  ✓ Exported via Spark (245.32 MB, 41,657 rows)

[2/4] Export RES view
  Table: building_enrichment_tsi_proportions_ind_res_view
  ✓ Exported via Spark (12.45 MB, 41,657 rows)

[3/4] Export COM view
  Table: building_enrichment_tsi_proportions_ind_com_view
  ✓ Exported via Spark (11.89 MB, 41,657 rows)

[4/4] Export IND view
  Table: building_enrichment_tsi_proportions_ind_ind_view
  ✓ Exported via Spark (10.23 MB, 41,657 rows)

================================================================================
GENERATING COUNTRY SUMMARY EXCEL
================================================================================

Generating summary tables...
  ✓ Country-level aggregation
  ✓ Urban breakdown
  ✓ Storey distribution

Writing Excel file...
  ✓ Sheet: Country Summary
  ✓ Sheet: Urban Breakdown
  ✓ Sheet: Grid Sample

  ✓ Excel saved: .../building_summary_country_layout_IND.xlsx

================================================================================
EXPORT COMPLETE
================================================================================

✅ Exported Files:
   • building_enrichment_output_IND_FULL.csv
   • building_enrichment_tsi_proportions_IND_RES_FULL.csv
   • building_enrichment_tsi_proportions_IND_COM_FULL.csv
   • building_enrichment_tsi_proportions_IND_IND_FULL.csv
   • building_summary_country_layout_IND.xlsx

📊 Total Export Size: 279.89 MB
```

---

## Complete Workflow

### End-to-End Example: Processing India

#### Step 1: Prepare Input Files

```bash
# Directory structure
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/
├── inputs/
│   ├── multipliers/
│   │   ├── IND_proportions.csv
│   │   └── tsi.csv
│   └── admin/
│       └── RMS_Admin0_geozones.gpkg
```

#### Step 2: Configure

Edit `config.yaml`:

```yaml
country:
  iso3: IND

inputs:
  proportions_csv: /Workspace/Users/.../IND_NOS_storey_mapping_041125.csv
  tsi_csv: /Volumes/.../inputs/multipliers/tsi.csv
  admin_boundaries: /Volumes/.../inputs/admin/RMS_Admin0_geozones.gpkg
  tile_footprint: /Volumes/.../inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp
```

Generate config:

```bash
python config_builder.py config.yaml --validate
```

#### Step 3: Run Pipeline

**Option A: Run all tasks sequentially**

```bash
#!/bin/bash
# run_pipeline.sh

CONFIG="config.json"

echo "Starting pipeline for $(jq -r '.iso3' $CONFIG)..."

echo "[1/7] Loading multipliers..."
python task1_proportions_to_delta.py --config_path $CONFIG || exit 1

echo "[2/7] Generating grid..."
python task2_grid_generation.py --config_path $CONFIG || exit 1

echo "[3/7] Downloading tiles..."
python task3_tile_downloader.py --config_path $CONFIG || exit 1

echo "[4/7] Extracting raster stats..."
python task4_raster_stats.py --config_path $CONFIG || exit 1

echo "[5/7] Post-processing..."
python task5_post_processing.py --config_path $CONFIG || exit 1

echo "[6/7] Creating views..."
python task6_create_views.py --config_path $CONFIG || exit 1

echo "[7/7] Exporting results..."
python task7_export.py --config_path $CONFIG || exit 1

echo "✅ Pipeline complete!"
```

**Option B: Databricks Workflow**

Configure as Databricks Job with 7 sequential tasks.

#### Step 4: Verify Results

**Check Delta Tables:**

```sql
-- Final output
SELECT COUNT(*) FROM prp_mr_bdap_projects.geospatialsolutions.building_enrichment_output;
-- Expected: ~40,000+ rows

-- Grid centroids
SELECT COUNT(*) FROM prp_mr_bdap_projects.geospatialsolutions.grid_centroids;
-- Expected: ~45,000+ rows

-- Download status
SELECT * FROM prp_mr_bdap_projects.geospatialsolutions.download_status;
-- Expected: All status = 'success'
```

**Check Exports:**

```bash
ls -lh /Volumes/.../outputs/exports/FULL_IND/
# Expected:
# building_enrichment_output_IND_FULL.csv (245 MB)
# building_enrichment_tsi_proportions_IND_RES_FULL.csv (12 MB)
# building_enrichment_tsi_proportions_IND_COM_FULL.csv (11 MB)
# building_enrichment_tsi_proportions_IND_IND_FULL.csv (10 MB)

ls -lh /Volumes/.../outputs/exports/
# Expected:
# building_summary_country_layout_IND.xlsx
```

#### Step 5: Analyze Results

Open Excel summary:

**Country Summary Sheet:**
```
Building Type | Storey 1 | Storey 2 | Storey 3 | ... | Total
--------------|----------|----------|----------|-----|-------
RES           | 45,234   | 38,921   | 29,456   | ... | 234,567
COM           | 12,456   | 10,234   | 8,123    | ... | 56,789
IND           | 8,234    | 6,789    | 5,123    | ... | 34,567
```

**Urban Breakdown:**
```
Urban Type | RES Buildings | COM Buildings | IND Buildings
-----------|---------------|---------------|---------------
Rural      | 45,234        | 5,678         | 2,345
Urban      | 156,789       | 38,901        | 28,765
Suburban   | 32,544        | 12,210        | 3,457
```

---

## Data Flow

### Detailed Data Lineage

```
INPUT FILES
├── proportions_csv ─────────┐
│                            │
├── tsi_csv ─────────────────┤
│                            ▼
│                      [Task 1: Load]
│                            │
│                            ▼
│                    Delta Tables:
│                    ├── proportions_table
│                    └── tsi_table
│                            │
├── admin_boundaries ────────┤
│                            │
└── tile_footprint ──────────┤
                             ▼
                       [Task 2: Grid]
                             │
                             ▼
                    Delta Table:
                    └── grid_centroids ────┐
                             │             │
                             ▼             │
                    [Task 3: Download]     │
                             │             │
                             ▼             │
                      GHSL Tiles           │
                      ├── built_c/         │
                      └── smod/            │
                             │             │
                             ├─────────────┘
                             ▼
                    [Task 4: Raster Stats]
                             │
                             ▼
                    Delta Table:
                    └── grid_counts
                             │
                   ┌─────────┴─────────┐
                   │                   │
            proportions_table     tsi_table
                   │                   │
                   └─────────┬─────────┘
                             ▼
                    [Task 5: Processing]
                             │
                             ▼
                    Delta Table:
                    └── building_enrichment_output
                             │
                             ▼
                    [Task 6: Create Views]
                             │
                             ▼
                        SQL Views:
                        ├── tsi_proportions_res_view
                        ├── tsi_proportions_com_view
                        └── tsi_proportions_ind_view
                             │
                             ▼
                    [Task 7: Export]
                             │
                             ▼
                      OUTPUT FILES
                      ├── CSV Exports (4 files)
                      └── Excel Summary (1 file)
```

### Data Transformations

**Task 1 → Task 2:**
- Proportions table used to validate grid generation
- No direct data flow

**Task 2 → Task 3:**
- `grid_centroids.tile_id` → List of tiles to download

**Task 3 → Task 4:**
- Downloaded tiles → Raster data source

**Task 2 → Task 4:**
- `grid_centroids` → Point locations for extraction

**Task 4 → Task 5:**
- `grid_counts` → Raw building counts

**Task 1 → Task 5:**
- `proportions_table` → Storey distribution
- `tsi_table` → TSI multipliers

**Task 5 → Task 6:**
- `building_enrichment_output` → Base table for views

**Task 5 & Task 6 → Task 7:**
- Output table + Views → CSV exports + Excel summary

---

## Troubleshooting

### Common Issues

#### 1. Config Generation Fails

**Error:**
```
KeyError: 'volume_root'
```

**Solution:**
Check `config.yaml` has all required sections:
```yaml
project:
  volume_root: /path/to/volume
```

#### 2. Task 1: CSV Format Issues

**Error:**
```
ValueError: Row sums do not equal 1.0
```

**Solution:**
- Check if values are percentages (sum to 100)
- Enable auto-normalization: `--auto_normalize_bad_rows True`

**Error:**
```
Could not parse numeric values
```

**Solution:**
- Check for mixed decimal formats
- Script auto-detects, but verify CSV encoding (UTF-8)

#### 3. Task 2: No Grids Generated

**Error:**
```
Generated 0 grid centroids
```

**Solution:**
- Verify `iso3` matches admin_boundaries ISO3 field
- Check tile footprint intersects country
- Verify CRS of input files

#### 4. Task 3: Download Failures

**Error:**
```
HTTP 404: Tile not found
```

**Solution:**
- Check tile_id format (should be like "R10_C15")
- Verify tile exists in GHSL repository
- Check internet connectivity

**Error:**
```
Extract failed: Zip file corrupted
```

**Solution:**
- Delete partial download
- Re-run with `--download_retries 3`

#### 5. Task 4: Out of Memory

**Error:**
```
OutOfMemoryError: Java heap space
```

**Solution:**
- Reduce `chunk_size` (try 5000 instead of 10000)
- Reduce `max_workers` (try 4 instead of 8)
- Reduce `tile_parallelism` (try 2 instead of 4)

#### 6. Task 5: Missing Proportions

**Error:**
```
Could not join proportions for built=11
```

**Solution:**
- Verify proportions_table has all built codes (11-15, 21-25)
- Check Task 1 completed successfully

#### 7. Task 6: VOID Column Error

**Error:**
```
NullType columns cannot be exported to CSV
```

**Solution:**
- This should not occur with current version
- If it does, check Task 6 uses `CAST(NULL AS <type>)`

#### 8. Task 7: Export Fails

**Error:**
```
Permission denied writing to /Volumes/...
```

**Solution:**
- Check write permissions to output_dir
- Verify volume is mounted in Databricks

### Debug Mode

Enable verbose logging:

```bash
# Add to any task
python task1_proportions_to_delta.py --config_path config.json --log_level DEBUG
```

View task execution plan:

```bash
# Preview without executing
python task2_grid_generation.py --config_path config.json --dry_run True
```

---

## Advanced Usage

### Processing Multiple Countries

**Create country-specific configs:**

```bash
# config_ind.yaml
country:
  iso3: IND

# config_usa.yaml
country:
  iso3: USA

# Generate configs
python config_builder.py config_ind.yaml --output config_ind.json
python config_builder.py config_usa.yaml --output config_usa.json

# Run pipelines
./run_pipeline.sh config_ind.json
./run_pipeline.sh config_usa.json
```

### Incremental Updates

**Re-run only specific tasks:**

```bash
# New proportions data? Re-run from Task 1
python task1_proportions_to_delta.py --config_path config.json
python task5_post_processing.py --config_path config.json
python task6_create_views.py --config_path config.json
python task7_export.py --config_path config.json

# New GHSL data released? Re-run from Task 3
python task3_tile_downloader.py --config_path config.json
python task4_raster_stats.py --config_path config.json
python task5_post_processing.py --config_path config.json
python task6_create_views.py --config_path config.json
python task7_export.py --config_path config.json
```

### Custom Grid Resolution

**Change from 2km to 5km:**

```yaml
params:
  cell_size: 5000  # Change from 2000 to 5000
```

Regenerate config and re-run from Task 2.

### Performance Tuning

**For smaller countries (< 10 tiles):**

```yaml
params:
  tile_parallelism: 2
  max_workers: 4
  chunk_size: 5000
```

**For larger countries (> 50 tiles):**

```yaml
params:
  tile_parallelism: 8
  max_workers: 12
  chunk_size: 20000
  download_concurrency: 5
```

### Custom Storey Bins

**Modify proportions mapping:**

Edit Task 5 to change storey bins (requires code modification).

### Export Subsets

**Export only specific LOBs:**

Edit Task 7:

```python
LOBS = ["res"]  # Only export RES, skip COM/IND
```

### Databricks Workflow Configuration

**Job JSON:**

```json
{
  "name": "GHSL Processing - IND",
  "tasks": [
    {
      "task_key": "task1_load",
      "python_wheel_task": {
        "entry_point": "task1_proportions_to_delta",
        "parameters": ["--config_path", "dbfs:/configs/config_ind.json"]
      },
      "cluster": {"existing_cluster_id": "..."}
    },
    {
      "task_key": "task2_grid",
      "depends_on": [{"task_key": "task1_load"}],
      "python_wheel_task": {
        "entry_point": "task2_grid_generation",
        "parameters": ["--config_path", "dbfs:/configs/config_ind.json"]
      },
      "cluster": {"existing_cluster_id": "..."}
    },
    ...
  ]
}
```

---

## FAQ

**Q: Can I process multiple countries in parallel?**
A: Yes, use separate configs and run as independent jobs.

**Q: How long does the full pipeline take?**
A: Depends on country size. India (~45k grids, 12 tiles): ~30-45 minutes.

**Q: Can I use different proportions for RES vs COM?**
A: Yes, proportions CSV has separate rows for each built code.

**Q: What if a tile download fails?**
A: Task 3 retries automatically. Check download_status table for failures.

**Q: Can I run on non-Databricks Spark?**
A: Yes, but you'll need to adapt volume paths to your file system.

**Q: How do I validate results?**
A: Check block SUMs in output (should match totals), review Excel summary.

**Q: Can I change the grid origin?**
A: Grid is based on Mollweide projection, origin is global. Cannot change.

**Q: What CRS should my admin boundaries be in?**
A: Any CRS - script reprojects to Mollweide automatically.

---

## Summary

This pipeline provides a **complete, automated solution** for generating building density estimates from GHSL data:

✅ **Easy configuration** - 20-line YAML generates 50+ config values
✅ **Comprehensive documentation** - Every task fully documented
✅ **Scalable** - Parallel processing with configurable workers
✅ **Reproducible** - Stable grid IDs and deterministic results
✅ **Flexible** - Works for any country, any resolution
✅ **Production-ready** - Error handling, retries, validation

**Next Steps:**
1. Configure for your country (`config.yaml`)
2. Generate config (`config_builder.py`)
3. Run pipeline (7 tasks sequentially)
4. Analyze results (CSV + Excel exports)

For detailed documentation on specific topics:
- Configuration: See `CONFIG_GUIDE.md`
- Architecture: See `README.md`
- Code details: See individual task docstrings

---

**Questions or issues?** Check the troubleshooting section or review task-specific documentation.

**Happy processing! 🚀**
