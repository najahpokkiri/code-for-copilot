# Data Folder

This folder contains the input data files required for the building enrichment pipeline.

## Required Files

Place the following files in this folder:

### 1. `NOS_storey_mapping.csv` **(YOU MUST PROVIDE THIS)**
Your country-specific NOS (Number of Stories) to storey proportion mapping file.

**Expected format:**
```csv
NOS,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8
1,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0
3,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0
...
```

- `NOS`: Number of stories value
- `P_1` to `P_8`: Proportions for each storey (must sum to 1.0)

### 2. `tsi.csv` (Provided)
TSI (Total Sum Insured) multipliers for different building types.

### 3. `RMS_Admin0_geozones.json.gz` (Provided)
Administrative boundaries file (GeoJSON compressed). GeoPandas can read this directly.

## Already Included

- `ghsl2_0_mwd_l1_tile_schema_land.gpkg` - GHSL tile footprints (used by all countries)

## Usage

In the notebook, these files are referenced as:
```python
PROPORTIONS_CSV = f"{WORKSPACE_BASE}/data/NOS_storey_mapping.csv"
TSI_CSV = f"{WORKSPACE_BASE}/data/tsi.csv"
ADMIN_BOUNDARIES = f"{WORKSPACE_BASE}/data/RMS_Admin0_geozones.json.gz"
```

**To process a new country:**
1. Replace `NOS_storey_mapping.csv` with your country-specific file
2. Update `ISO3` in the notebook
3. Run all cells
