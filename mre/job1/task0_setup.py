#!/usr/bin/env python3
"""
Task 0 — Environment Setup & Config Generation

Sets up the environment and generates full configuration.

This task:
1. Reads minimal config.yaml (user inputs)
2. Uses config_builder.py to generate full config.json
3. Creates {ISO3}/ folder structure
4. Copies input files to correct locations
5. Saves full config.json to {ISO3}/config.json for Task 1-7

Usage:
------
  python task0_setup.py --minimal_config /path/to/minimal_config.yaml

Output:
-------
  - Folder structure: {ISO3}/input/, {ISO3}/output/, {ISO3}/logs/
  - Full config.json at {ISO3}/config.json
  - Copied files in correct locations
"""

import os
import sys
import json
import inspect
import yaml
from pathlib import Path

# Add current directory to path to import config_builder
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path(inspect.stack()[0].filename).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from config_builder import ConfigBuilder

# ================================================================================
# CLI ARGUMENT PARSER
# ================================================================================
minimal_config_path = None
if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv):
        if arg == "--minimal_config" and i + 1 < len(sys.argv):
            minimal_config_path = sys.argv[i + 1]
            break

if not minimal_config_path:
    print("❌ Error: --minimal_config required")
    sys.exit(1)

print("="*80)
print("TASK 0: ENVIRONMENT SETUP & CONFIG GENERATION")
print("="*80)
print(f"Minimal config: {minimal_config_path}")
print("="*80)

# ================================================================================
# STEP 1: LOAD MINIMAL CONFIG
# ================================================================================
print("\n📄 Loading minimal configuration...")

# Handle dbfs: paths
config_path_local = minimal_config_path
if minimal_config_path.startswith("dbfs:"):
    config_path_local = minimal_config_path.replace("dbfs:", "/dbfs")

with open(config_path_local, 'r') as f:
    minimal_config = yaml.safe_load(f)

ISO3 = minimal_config['country']['iso3']
VOLUME_ROOT = minimal_config['project']['volume_root']

print(f"  ✓ ISO3: {ISO3}")
print(f"  ✓ Volume root: {VOLUME_ROOT}")

# Initialize Spark and dbutils
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
except:
    import IPython
    dbutils = IPython.get_ipython().user_ns.get('dbutils')

# ================================================================================
# STEP 2: CREATE FOLDER STRUCTURE
# ================================================================================
print("\n📁 Creating ISO3-based folder structure...")

BASE_DATA_DIR = f"{VOLUME_ROOT}/{ISO3}"
INPUT_DIR = f"{BASE_DATA_DIR}/input"
TILES_DIR = f"{INPUT_DIR}/tiles"
OUTPUT_DIR = f"{BASE_DATA_DIR}/output"
LOGS_DIR = f"{BASE_DATA_DIR}/logs"

for folder in [BASE_DATA_DIR, INPUT_DIR, TILES_DIR, f"{TILES_DIR}/built_c", f"{TILES_DIR}/smod", OUTPUT_DIR, LOGS_DIR]:
    dbutils.fs.mkdirs(folder)
    print(f"  ✓ {folder}")

# Update minimal config with ISO3-based paths
minimal_config['project']['volume_root'] = BASE_DATA_DIR

# ================================================================================
# STEP 3: COPY INPUT FILES
# ================================================================================
print("\n📂 Copying input files to {ISO3}/input/...")

# Get workspace base for tile footprint
workspace_base = minimal_config.get('workspace_base', '/Workspace/Users/npokkiri@munichre.com/inventory_nos_db/code-for-copilot-main/mre/job1')

# Copy tile footprint
tile_source = f"{workspace_base}/data/ghsl2_0_mwd_l1_tile_schema_land.gpkg"
tile_dest = f"{TILES_DIR}/ghsl2_0_mwd_l1_tile_schema_land.gpkg"

try:
    if tile_source.startswith("/Workspace"):
        tile_src_uri = f"file:{tile_source}"
    else:
        tile_src_uri = tile_source
    dbutils.fs.cp(tile_src_uri, tile_dest, recurse=True)
    print(f"  ✓ Tiles: {tile_source} → {tile_dest}")
    minimal_config['inputs']['tile_footprint'] = tile_dest
except Exception as e:
    print(f"  ⚠️  Tile copy failed: {e}, using source path")
    minimal_config['inputs']['tile_footprint'] = tile_source

# Copy TSI CSV
tsi_source = minimal_config['inputs'].get('tsi_csv')
if tsi_source:
    tsi_dest = f"{INPUT_DIR}/tsi.csv"
    try:
        dbutils.fs.cp(tsi_source, tsi_dest, recurse=True)
        print(f"  ✓ TSI: {tsi_source} → {tsi_dest}")
        minimal_config['inputs']['tsi_csv'] = tsi_dest
    except Exception as e:
        print(f"  ⚠️  TSI copy failed: {e}, using source path")

# Copy Proportions CSV
prop_source = minimal_config['inputs'].get('proportions_csv')
if prop_source:
    prop_dest = f"{INPUT_DIR}/proportions.csv"
    try:
        dbutils.fs.cp(prop_source, prop_dest, recurse=True)
        print(f"  ✓ Proportions: {prop_source} → {prop_dest}")
        minimal_config['inputs']['proportions_csv'] = prop_dest
    except Exception as e:
        print(f"  ⚠️  Proportions copy failed: {e}, using source path")

# Copy admin boundaries (optional)
admin_source = minimal_config['inputs'].get('admin_boundaries')
if admin_source and admin_source not in ['', 'None', None]:
    admin_dest = f"{INPUT_DIR}/admin_boundaries.gpkg"
    try:
        dbutils.fs.cp(admin_source, admin_dest, recurse=True)
        print(f"  ✓ Admin boundaries: {admin_source} → {admin_dest}")
        minimal_config['inputs']['admin_boundaries'] = admin_dest
    except Exception as e:
        print(f"  ⚠️  Admin boundaries copy failed: {e}, using source path")

# ================================================================================
# STEP 4: GENERATE FULL CONFIG.JSON USING CONFIG_BUILDER
# ================================================================================
print("\n⚙️  Generating full config.json using config_builder.py...")

# Save updated minimal config to temp file
temp_yaml_path = f"/tmp/minimal_config_{ISO3}.yaml"
with open(temp_yaml_path, 'w') as f:
    yaml.dump(minimal_config, f)

# Use ConfigBuilder to generate full config
try:
    builder = ConfigBuilder(temp_yaml_path)
    full_config = builder.build_full_config()
    print(f"  ✓ Generated full configuration with ISO3 suffix for all tables")
    print(f"  ✓ Example: building_enrichment_output_{ISO3}")
except Exception as e:
    print(f"  ❌ Config generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ================================================================================
# STEP 5: SAVE FULL CONFIG TO {ISO3}/config.json
# ================================================================================
print("\n💾 Saving full config.json...")

config_dest = f"{BASE_DATA_DIR}/config.json"
config_dest_local = config_dest.replace('dbfs:', '/dbfs').replace('/Volumes', '/Volumes')

# Ensure directory exists
os.makedirs(os.path.dirname(config_dest_local), exist_ok=True)

# Save full config
with open(config_dest_local, 'w') as f:
    json.dump(full_config, f, indent=2)

print(f"  ✓ Full config saved to: {config_dest}")
print(f"  ✓ This config will be used by Task 1-7")

# ================================================================================
# SUMMARY
# ================================================================================
print("\n" + "="*80)
print("SETUP COMPLETE")
print("="*80)
print(f"Country: {ISO3}")
print(f"Base directory: {BASE_DATA_DIR}")
print(f"Config location: {config_dest}")
print(f"")
print(f"Key table names (with ISO3 suffix):")
print(f"  - {full_config.get('output_table')}")
print(f"  - {full_config.get('grid_source')}")
print(f"  - {full_config.get('counts_delta_table')}")
print("="*80)

print("\n✅ Task 0 completed successfully")
print(f"📌 Next tasks will use config at: {config_dest}")

# Clean up temp file
try:
    os.remove(temp_yaml_path)
except:
    pass
