# Reference Data

Place small reference datasets here (e.g., tile footprints, small shapefiles).

## ⚠️ Important: File Size Limits

**This directory is for SMALL reference files only:**

| File Type | Max Size | Example |
|-----------|----------|---------|
| **Tile footprints** | < 50 MB | ✅ GHSL tile schema |
| **Small shapefiles** | < 50 MB | ✅ Region boundaries |
| **World shapefiles** | ANY | ❌ TOO LARGE - Use Volumes! |

## 📂 Subdirectories

### tile_footprint/

Place GHSL tile footprint shapefile here.

**Expected files**:
```
tile_footprint/
├── GHSL2_0_MWD_L1_tile_schema_land.gpkg
├── GHSL2_0_MWD_L1_tile_schema_land.shx
├── GHSL2_0_MWD_L1_tile_schema_land.dbf
├── GHSL2_0_MWD_L1_tile_schema_land.prj
└── GHSL2_0_MWD_L1_tile_schema_land.cpg (optional)
```

**Size check**:
```bash
du -sh tile_footprint/
# Should be < 50 MB
```

**If too large**: Keep in Databricks Volumes instead!

## 📥 How to Add Tile Footprint

```bash
# Copy GeoPackage file (single file - easier than shapefile!)
cp /path/to/GHSL2_0_MWD_L1_tile_schema_land.gpkg tile_footprint/

# Verify file is present
ls -lh tile_footprint/*.gpkg
```

## ⚙️ Configuration

In `config.yaml`:

```yaml
inputs:
  # ✅ If bundled (small file):
  tile_footprint: ${workspace.root_path}/files/data/inputs/reference_data/tile_footprint/GHSL2_0_MWD_L1_tile_schema_land.gpkg

  # ❌ If too large (use Volumes):
  tile_footprint: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/reference_data/tiles/GHSL2_0_MWD_L1_tile_schema_land.gpkg
```

## 🚫 What NOT to Include Here

### World Administrative Boundaries (RMS_Admin0_geozones.gpkg)

❌ **DO NOT place the world shapefile here!**

**Why?**
- File size: ~250 MB (too large for git)
- Binary file (causes git bloat)
- Static reference (doesn't change per analysis)
- Shared across all projects

**Where to keep it:**
```
Databricks Volumes:
/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg
```

**In config.yaml:**
```yaml
inputs:
  admin_boundaries: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/admin/RMS_Admin0_geozones.gpkg
```

## 📏 File Size Guidelines

### ✅ Include in Bundle

- Small shapefiles (< 50 MB)
- Tile footprints (< 50 MB)
- Grid templates (< 20 MB)
- Lookup tables (< 10 MB)

### ❌ Keep in Volumes

- World boundaries (250 MB) ← **Your case**
- Large raster datasets
- Downloaded tiles
- Any file > 100 MB

## 🔍 Checking File Sizes

```bash
# Check GeoPackage file size
ls -lh tile_footprint/*.gpkg

# Check entire directory
du -sh tile_footprint/

# If > 50 MB, move to Volumes!
```

## 💡 Best Practices

1. **Document source**: Add a `tile_footprint/SOURCE.txt`:
   ```
   File: GHSL2_0_MWD_L1_tile_schema_land.gpkg
   Source: JRC GHSL Data Portal
   Download date: 2024-01-15
   URL: https://ghsl.jrc.ec.europa.eu/
   ```

2. **Version reference data**: If the schema updates:
   ```
   reference_data/
   └── tile_footprint/
       ├── GHSL_tile_schema_2023.gpkg  # Previous version
       └── GHSL_tile_schema_2024.gpkg  # Current version
   ```

3. **Test with small subset**: For development, create a small test version:
   ```
   tile_footprint/
   ├── GHSL_tile_schema_land.gpkg       # Full version
   └── GHSL_tile_schema_land_test.gpkg  # Subset for testing
   ```

## ❓ FAQ

**Q: My tile footprint is 75 MB - too large?**
A: Yes! Move it to Volumes. Update `config.yaml` to reference the Volume path.

**Q: Can I add other reference GeoPackages or shapefiles?**
A: Yes, create subdirectories:
```
reference_data/
├── tile_footprint/          # .gpkg files
├── coastal_zones/           # .gpkg or .shp
└── urban_extents/           # .gpkg or .shp
```

Just keep each < 50 MB! (GeoPackage format preferred - single file is easier)

**Q: The world shapefile is critical - are you sure I can't bundle it?**
A: Correct - at 250 MB it will cause issues:
- Git becomes slow
- Clone times increase dramatically
- Binary diffs don't work
- Wastes storage

Keep it in Volumes where large files belong!
