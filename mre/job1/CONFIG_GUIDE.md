# Configuration System Guide

## 📋 Overview

The pipeline now uses a **simplified YAML-based configuration system** that auto-generates all derived paths and table names. This reduces configuration from 54 lines to ~20 lines of essential settings.

### Benefits

✅ **Less verbose**: Specify only root paths, ISO code, and key parameters
✅ **Auto-generation**: Automatically creates all derived paths, table names, folders
✅ **Type-safe**: Prevents typos in repeated path patterns
✅ **Easy to modify**: Change one country? Just update ISO3 and regenerate
✅ **Validated**: Built-in comparison with existing configs

---

## 🚀 Quick Start

### 1. Edit the YAML configuration

```bash
vim config.yaml
```

Update the essential values:

```yaml
country:
  iso3: USA  # Change country here

inputs:
  proportions_csv: /path/to/USA_proportions.csv
  # Update other input file paths...
```

### 2. Generate the full configuration

```bash
python config_builder.py config.yaml
```

This creates `config.json` with all 50+ derived values.

### 3. Use in your pipeline

All task scripts continue to use `config.json` as before:

```bash
python task1_proportions_to_delta.py --config_path config.json
```

---

## 📁 File Structure

```
mre/job1/
├── config.yaml              # ⭐ EDIT THIS (simplified, ~20 lines)
├── config_builder.py        # Builder script (run to generate)
├── config.json              # Auto-generated (don't edit manually)
├── config.json.backup       # Backup of original config
├── task1_proportions_to_delta.py
├── task2_grid_generation.py
└── ...
```

---

## 🔧 Configuration Structure

### config.yaml (Simplified)

```yaml
# ============================================================================
# PROJECT SETTINGS
# ============================================================================
project:
  catalog: prp_mr_bdap_projects              # Databricks catalog
  schema: geospatialsolutions                # Databricks schema
  volume_root: /Volumes/.../data             # Base volume path

# ============================================================================
# COUNTRY SETTINGS
# ============================================================================
country:
  iso3: IND  # 🌍 Change this for different countries (USA, BRA, etc.)

# ============================================================================
# INPUT FILES (User-provided paths)
# ============================================================================
inputs:
  proportions_csv: /path/to/proportions.csv
  tsi_csv: /path/to/tsi.csv
  admin_boundaries: /path/to/boundaries.gpkg
  tile_footprint: /path/to/tile_schema.shp

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
params:
  cell_size: 2000                   # Grid cell size (meters)
  datasets: built_c,smod            # GHSL datasets to download
  chunk_size: 10000                 # Processing chunk size
  max_workers: 8                    # Parallel workers
  # ... more parameters ...
```

### What Gets Auto-Generated

From the simple YAML above, the builder creates:

#### 🗂️ Folder Paths
```json
{
  "tiles_dest_root": "/Volumes/.../data/inputs/tiles",
  "built_root": "/Volumes/.../data/inputs/tiles/built_c",
  "smod_root": "/Volumes/.../data/inputs/tiles/smod",
  "output_dir": "/Volumes/.../data/outputs",
  "grid_output_csv": "/Volumes/.../data/outputs/grid_centroids.csv"
}
```

#### 📊 Table Names
```json
{
  "proportions_table": "catalog.schema.building_enrichment_proportions_input",
  "tsi_table": "catalog.schema.building_enrichment_tsi_input",
  "grid_source": "catalog.schema.grid_centroids",
  "download_status_table": "catalog.schema.download_status",
  "counts_delta_table": "catalog.schema.grid_counts",
  "output_table": "catalog.schema.building_enrichment_output"
}
```

#### ⚙️ Admin Fields
```json
{
  "admin_field": "ISO3",
  "admin_value": "IND",
  "tile_id_field": "tile_id"
}
```

---

## 💡 Common Use Cases

### Switching to a New Country

**Before** (54-line config.json):
- Find and replace ~15 instances of "IND" → "USA"
- Update ~10 path references
- Update ~7 table name suffixes
- Risk: Missing a reference causes runtime errors

**Now** (20-line config.yaml):
```bash
# 1. Edit YAML
sed -i 's/iso3: IND/iso3: USA/' config.yaml
sed -i 's/IND_NOS/USA_NOS/' config.yaml

# 2. Regenerate
python config_builder.py config.yaml

# Done! All 50+ values updated consistently.
```

### Testing with a Different Volume Root

```yaml
project:
  volume_root: /Volumes/test_environment/data  # Test environment
```

Regenerate → All paths update automatically.

### Changing Grid Resolution

```yaml
params:
  cell_size: 5000  # Change from 2km to 5km
```

Regenerate → Config updated, paths remain consistent.

---

## 🛠️ Advanced Usage

### View Summary Without Writing Files

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

📋 Delta Tables Generated:
   • proportions_table: prp_mr_bdap_projects.geospatialsolutions.building_enrichment_proportions_input
   • tsi_table: prp_mr_bdap_projects.geospatialsolutions.building_enrichment_tsi_input
   ...

🗂️  Folder Structure:
   • /Volumes/.../data/inputs/tiles/built_c
   • /Volumes/.../data/inputs/tiles/smod
   ...
```

### Validate Against Existing Config

```bash
python config_builder.py config.yaml --validate
```

Shows differences between generated and existing configs.

### Write to Custom Output Path

```bash
python config_builder.py config.yaml --output config_usa.json
```

---

## 🔍 Comparison: Before vs After

### Before (config.json - 54 lines)

```json
{
  "catalog": "prp_mr_bdap_projects",
  "schema": "geospatialsolutions",
  "proportions_csv_path": "/Workspace/Users/.../IND_NOS_storey_mapping_041125.csv",
  "tsi_csv_path": "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/multipliers/tsi.csv",
  "csv_infer_schema": true,
  "proportions_path": "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_proportions_input",
  "iso3": "IND",
  "grid_output_csv": "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/output/grid_centroids.csv",
  "delta_table_base": "prp_mr_bdap_projects.geospatialsolutions.grid_centroids",
  "cell_size": 2000,
  "export_crs": "EPSG:4326",
  "dry_run": false,
  "tiles_dest_root": "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles",
  "download_status_table": "prp_mr_bdap_projects.geospatialsolutions.download_status",
  "datasets": "built_c,smod",
  "download_concurrency": 3,
  "download_retries": 2,
  "spark_tmp_dir": "/tmp/job3_grid_tmp",
  "tile_parallelism": "4",
  "SAMPLE_SIZE": 10000,
  "overwrite_schema": true,
  "preview": true,
  "preview_rows": 5,
  "grid_source": "prp_mr_bdap_projects.geospatialsolutions.grid_centroids",
  "built_root": "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/built_c",
  "smod_root": "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/inputs/tiles/smod",
  "output_dir": "/Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data/outputs",
  ...
}
```

**Issues**:
- 🔴 54 lines with repetitive paths
- 🔴 Volume root repeated 10+ times
- 🔴 Table naming pattern repeated 7 times
- 🔴 ISO3 appears in 3 different places
- 🔴 Easy to create inconsistencies

### After (config.yaml - ~20 lines)

```yaml
project:
  catalog: prp_mr_bdap_projects
  schema: geospatialsolutions
  volume_root: /Volumes/prp_mr_bdap_projects/geospatialsolutions/external/jrc/data

country:
  iso3: IND

inputs:
  proportions_csv: /Workspace/Users/.../IND_NOS_storey_mapping_041125.csv
  tsi_csv: /Volumes/.../data/inputs/multipliers/tsi.csv
  admin_boundaries: /Volumes/.../data/inputs/admin/RMS_Admin0_geozones.gpkg
  tile_footprint: /Volumes/.../data/inputs/tiles/GHSL2_0_MWD_L1_tile_schema_land.shp

params:
  cell_size: 2000
  datasets: built_c,smod
  chunk_size: 10000
  max_workers: 8
```

**Benefits**:
- ✅ 20 lines (63% reduction)
- ✅ Volume root defined once
- ✅ ISO3 defined once
- ✅ No repetitive patterns
- ✅ Auto-generates consistent paths

---

## 🔄 Migration Guide

### For New Jobs

1. Copy `config.yaml` and `config_builder.py` to your job folder
2. Edit `config.yaml` with your settings
3. Run `python config_builder.py config.yaml`
4. Use generated `config.json` in your tasks

### For Existing Jobs

Your existing task scripts don't need changes! They still read `config.json`.

Just use the builder to generate it from YAML:

```bash
# Backup existing config
cp config.json config.json.old

# Generate from YAML
python config_builder.py config.yaml

# Validate
python config_builder.py config.yaml --validate
```

---

## 📚 Reference

### Builder Command Options

```bash
python config_builder.py <yaml_file> [OPTIONS]
```

**Options**:
- `--output FILE` - Write to custom file (default: config.json)
- `--validate` - Compare with existing config.json
- `--summary` - Show summary without writing files

### Folder Structure Created

```
{volume_root}/
├── inputs/
│   ├── tiles/
│   │   ├── built_c/         # GHSL built-up tiles
│   │   └── smod/            # GHSL settlement model tiles
│   ├── admin/               # Admin boundaries
│   └── multipliers/         # TSI and proportion CSVs
└── outputs/                 # Pipeline outputs
```

### Table Naming Convention

Pattern: `{catalog}.{schema}.{table_name}`

Example: `prp_mr_bdap_projects.geospatialsolutions.grid_centroids`

---

## ❓ FAQ

**Q: Do I need to modify my task scripts?**
A: No! Scripts still read `config.json`. Just use the builder to generate it.

**Q: What if I need a custom path not in the YAML?**
A: Either: (1) Add it to the YAML structure, or (2) Manually edit `config.json` after generation.

**Q: Can I use this with multiple countries?**
A: Yes! Create separate YAMLs: `config_IND.yaml`, `config_USA.yaml`, etc.

**Q: What if generation fails?**
A: Check YAML syntax. The builder validates all required fields.

**Q: Can I version control both YAML and JSON?**
A: Commit YAML to git. Auto-generate JSON as needed. (Or commit both for reference.)

---

## 🎯 Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of config** | 54 | ~20 |
| **Repetitive paths** | Many | None |
| **Table name patterns** | Manual | Auto-generated |
| **Country switching** | Error-prone | Change 1 line |
| **Validation** | Manual | Built-in |

**Result**: 63% less configuration, 100% more maintainable.

---

## 📞 Support

For issues or questions:
1. Check this guide
2. Run with `--validate` to compare configs
3. Review generated `config.json` structure
4. Check task script compatibility

---

**Happy configuring! 🚀**
