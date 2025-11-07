# Input Data Directory

This directory contains input files for the Building Data Enrichment pipeline.

## 📂 Directory Structure

```
data/inputs/
├── proportions/          # Building type proportions CSVs (per country)
│   ├── README.md        # Instructions
│   └── IND_NOS_storey_mapping.csv (place your file here)
│
├── multipliers/          # TSI multipliers and other multipliers
│   ├── README.md        # Instructions
│   └── tsi.csv (place your file here)
│
└── reference_data/       # Reference datasets (small files only)
    ├── README.md        # Instructions
    └── tile_footprint/
        └── GHSL2_0_MWD_L1_tile_schema_land.shp (place shapefiles here)
```

## 📋 What Goes Here

### ✅ Include in Bundle (Small Files)

**Proportions CSV** (`proportions/`)
- Building type distribution by storey
- One file per country (e.g., `IND_NOS_storey_mapping.csv`)
- Usually < 10 MB
- **Action**: Place your country-specific CSV file here

**TSI Multipliers** (`multipliers/`)
- Total Sum Insured multipliers
- Small CSV file (< 5 MB)
- **Action**: Place `tsi.csv` here

**Tile Footprint** (`reference_data/tile_footprint/`)
- GHSL tile schema shapefile
- Small shapefile (< 50 MB)
- Static reference data
- **Action**: Place shapefile and related files (.shp, .shx, .dbf, .prj) here

### ❌ DO NOT Include (Large Files → Use Volumes)

**World Administrative Boundaries**
- File: `RMS_Admin0_geozones.gpkg` (~250 MB)
- **Too large for git!**
- **Keep in Databricks Volumes**: `/Volumes/.../reference_data/admin/`

## 🚀 Setup Instructions for Analysts

### Step 1: Place Your Input Files

```bash
cd databricks_bundle_example/data/inputs/

# 1. Add your proportions CSV
cp /path/to/your/IND_NOS_storey_mapping.csv proportions/

# 2. Add TSI multipliers
cp /path/to/your/tsi.csv multipliers/

# 3. Add tile footprint shapefile (if you have it)
cp /path/to/GHSL2_0_MWD_L1_tile_schema_land.* reference_data/tile_footprint/
```

### Step 2: Update config.yaml

Edit the root `config.yaml` to reference your files:

```yaml
inputs:
  # ✅ Bundled files (small, in data/)
  proportions_csv: ${workspace.root_path}/files/data/inputs/proportions/IND_NOS_storey_mapping.csv
  tsi_csv: ${workspace.root_path}/files/data/inputs/multipliers/tsi.csv
  tile_footprint: ${workspace.root_path}/files/data/inputs/reference_data/tile_footprint/GHSL2_0_MWD_L1_tile_schema_land.shp

  # ❌ Large files (keep in Volumes)
  admin_boundaries: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg
```

### Step 3: Deploy Bundle

```bash
cd databricks_bundle_example/
databricks bundle deploy

# Your data files are automatically uploaded to workspace!
```

## 📏 File Size Guidelines

| File Type | Max Size | Location |
|-----------|----------|----------|
| **CSV files** | < 50 MB | ✅ Bundle (`data/inputs/`) |
| **Small shapefiles** | < 50 MB | ✅ Bundle (`data/inputs/reference_data/`) |
| **Large shapefiles** | > 50 MB | ❌ Volumes only |
| **GeoPackages** | Any size | ❌ Volumes only (unless tiny) |

## 🔍 Verifying Your Setup

After placing files:

```bash
# Check files are in place
ls -lh data/inputs/proportions/
ls -lh data/inputs/multipliers/
ls -lh data/inputs/reference_data/tile_footprint/

# Validate bundle
databricks bundle validate

# Deploy
databricks bundle deploy
```

## ❓ FAQ

**Q: What if my tile footprint is large (> 50 MB)?**
A: Keep it in Volumes. Update `config.yaml` to reference the Volume path instead.

**Q: Can I add multiple country proportions files?**
A: Yes! Place multiple files in `proportions/` directory:
```
proportions/
├── IND_NOS_storey_mapping.csv
├── USA_NOS_storey_mapping.csv
└── BRA_NOS_storey_mapping.csv
```

Then update `config.yaml` to reference the one you want to use.

**Q: Where is the world shapefile (RMS_Admin0_geozones.gpkg)?**
A: It's too large (250 MB) for git. Keep it in Databricks Volumes:
```
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/
```

**Q: Git says my file is too large?**
A: Check `.gitignore` - files > 100 MB are automatically blocked. Move large files to Volumes.

## 📖 Related Documentation

- See `../DATA_ORGANIZATION.md` for detailed guidance on data placement
- See `../STRUCTURE.md` for complete bundle structure
- See `README.md` for deployment instructions
