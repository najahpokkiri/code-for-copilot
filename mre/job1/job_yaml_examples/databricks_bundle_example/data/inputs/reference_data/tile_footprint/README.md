# Tile Footprint Shapefile

Place the GHSL tile footprint shapefile components here.

## 📁 Required Files

**All shapefile components must be present:**

```
tile_footprint/
├── GHSL2_0_MWD_L1_tile_schema_land.shp  ← Main shapefile
├── GHSL2_0_MWD_L1_tile_schema_land.shx  ← Shape index
├── GHSL2_0_MWD_L1_tile_schema_land.dbf  ← Attribute table
├── GHSL2_0_MWD_L1_tile_schema_land.prj  ← Projection info
└── GHSL2_0_MWD_L1_tile_schema_land.cpg  ← Character encoding (optional)
```

## 📥 Setup

1. **Copy all shapefile components**:
   ```bash
   cp /path/to/GHSL2_0_MWD_L1_tile_schema_land.* .
   ```

2. **Verify all files present**:
   ```bash
   ls -lh
   # Should show all 4-5 files
   ```

3. **Check file size**:
   ```bash
   du -sh .
   # Should be < 50 MB for bundling
   ```

## ⚙️ Configuration

Update `config.yaml` at bundle root:

```yaml
inputs:
  tile_footprint: ${workspace.root_path}/files/data/inputs/reference_data/tile_footprint/GHSL2_0_MWD_L1_tile_schema_land.shp
```

## 🔍 Data Info

**GHSL Tile Schema**:
- Source: JRC GHSL Data Portal
- Purpose: Defines the tiling scheme for GHSL raster datasets
- Used by: Task 2 (Grid Generation) to determine which tiles cover the study area

## ⚠️ Size Warning

If your shapefile is > 50 MB:
1. ❌ **Don't bundle it** (too large for git)
2. ✅ **Place in Volumes**: `/Volumes/.../reference_data/tiles/`
3. ✅ **Update config.yaml** to reference Volume path

## 💾 Placeholder

This directory is currently empty. **You need to add the shapefile components.**

If you don't have the file yet, it should be available from:
- Your Databricks Volumes
- Shared team storage
- Downloaded from JRC GHSL portal
