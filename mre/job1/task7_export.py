#!/usr/bin/env python3
"""
Export Building Enrichment FULL Datasets

Exports ALL ROWS from:
1. Main building enrichment output table
2. RES view (all rows)
3. COM view (all rows)
4. IND view (all rows)

Usage:
    python export_full_dataset.py --iso3 IND
"""

import os
import sys
import shutil
from datetime import datetime

from pyspark.sql import SparkSession
os.environ["CONFIG_PATH"] = "./config.json"
# Parse command line arguments
def parse_args():
    args = {"iso3": "IND"}
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv), 2):
            if sys.argv[i].startswith("--"):
                key = sys.argv[i].lstrip("--")
                if i + 1 < len(sys.argv):
                    args[key] = sys.argv[i + 1]
    return args

args = parse_args()
ISO3 = args["iso3"].upper()

# Configuration
CATALOG = "prp_mr_bdap_projects"
SCHEMA = "geospatialsolutions"
BASE_OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.building_enrichment_output_{ISO3.lower()}"
BASE_VIEW_NAME = f"{CATALOG}.{SCHEMA}.building_enrichment_tsi_proportions_{ISO3.lower()}"

# Export paths - save directly to Volumes as individual CSVs
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPORT_FOLDER = f"/Volumes/{CATALOG}/{SCHEMA}/external/jrc/data/outputs/exports/FULL_{ISO3}_{TIMESTAMP}"

# LOBs to export
LOBS = ["res", "com", "ind"]

print("="*80)
print("BUILDING ENRICHMENT FULL DATASET EXPORT")
print("="*80)
print(f"ISO3:          {ISO3}")
print(f"Export size:   FULL DATASET (ALL ROWS)")
print(f"Catalog:       {CATALOG}")
print(f"Schema:        {SCHEMA}")
print(f"Export folder: {EXPORT_FOLDER}")
print(f"Output format: Individual CSV files (4 total)")
print("="*80)

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Get dbutils
try:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
except:
    import IPython
    dbutils = IPython.get_ipython().user_ns.get('dbutils')

def export_full_table_to_csv(table_name, output_path, description):
    """
    Export FULL table to a single clean CSV file using Spark (not Pandas).
    Handles large datasets efficiently.
    
    Args:
        table_name: Fully qualified table name
        output_path: Full path including .csv filename (Volumes path)
        description: Human-readable description for logging
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{description}")
    print(f"  Table: {table_name}")
    print(f"  Output: {output_path}")
    
    try:
        # Check if table exists
        if not spark.catalog.tableExists(table_name):
            print(f"  ⚠️  WARNING: Table {table_name} does not exist. Skipping.")
            return False
            
        # Load full table
        df = spark.table(table_name)

        # Drop VOID columns (not supported by CSV writer)
        void_columns = [field.name for field in df.schema.fields if str(field.dataType).upper() == "VOIDTYPE()"]
        if void_columns:
            print(f"  Dropping VOID columns: {', '.join(void_columns)}")
            df = df.drop(*void_columns)

        row_count = df.count()
        print(f"  Total rows: {row_count:,}")
        
        if row_count == 0:
            print(f"  ⚠️  WARNING: Table is empty. Skipping.")
            return False
        
        print(f"  Exporting ALL {row_count:,} rows...")
        
        # Write to temp folder using Spark (handles large data efficiently)
        temp_folder = f"/tmp/spark_export_{os.path.basename(output_path).replace('.csv', '')}"
        
        print(f"  Writing to temporary folder...")
        # Coalesce to 1 file for single CSV output
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_folder)
        
        # Find the actual CSV file (Spark creates part-*.csv)
        print(f"  Finding CSV file...")
        csv_files = [f for f in os.listdir(temp_folder) if f.endswith(".csv") and not f.startswith("_")]
        
        if not csv_files:
            print(f"  ❌ ERROR: No CSV file found in temp folder")
            return False
        
        csv_path = os.path.join(temp_folder, csv_files[0])
        
        # Copy to Volumes using dbutils
        print(f"  Copying to Volumes...")
        dbutils.fs.cp(f"file:{csv_path}", output_path)
        
        # Get file size
        file_info = dbutils.fs.ls(output_path)
        file_size_mb = file_info[0].size / (1024 * 1024) if file_info else 0
        
        # Cleanup temp folder
        print(f"  Cleaning up temp files...")
        shutil.rmtree(temp_folder)
        
        print(f"  ✓ Export complete ({file_size_mb:.2f} MB, {row_count:,} rows)")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Create export folder
try:
    dbutils.fs.mkdirs(EXPORT_FOLDER)
    print(f"\n✓ Created export folder: {EXPORT_FOLDER}")
except Exception as e:
    print(f"\n✓ Using export folder: {EXPORT_FOLDER}")

print("\n" + "="*80)
print("EXPORTING FULL DATASETS")
print("="*80)

exported_files = []

# 1. Export main output table (FULL)
print("\n" + "-"*80)
main_csv = f"{EXPORT_FOLDER}/building_enrichment_output_{ISO3}_FULL.csv"
if export_full_table_to_csv(
    BASE_OUTPUT_TABLE, 
    main_csv,
    f"[1/4] Exporting FULL main output table"
):
    exported_files.append(main_csv)

# 2-4. Export LOB views (FULL)
for idx, lob in enumerate(LOBS, start=2):
    print("\n" + "-"*80)
    view_name = f"{BASE_VIEW_NAME}_{lob}_view"
    view_csv = f"{EXPORT_FOLDER}/building_enrichment_tsi_proportions_{ISO3}_{lob.upper()}_FULL.csv"
    
    if export_full_table_to_csv(
        view_name,
        view_csv,
        f"[{idx}/4] Exporting FULL {lob.upper()} view"
    ):
        exported_files.append(view_csv)

# Create README file
print("\n" + "="*80)
print("CREATING README")
print("="*80)

readme_content = f"""Building Enrichment FULL Dataset Export
=========================================

Export Details:
- ISO3 Code: {ISO3}
- Export Size: FULL DATASET (ALL ROWS)
- Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Catalog: {CATALOG}
- Schema: {SCHEMA}

Files Included:
---------------
1. building_enrichment_output_{ISO3}_FULL.csv
   - Main building enrichment output table (ALL ROWS)
   - Contains all building counts, storey distributions, TSI values, and percentages
   - Columns include: GRID_ID, lat, lon, urban, nr* columns, storey* columns, TSI columns, percentage columns

2. building_enrichment_tsi_proportions_{ISO3}_RES_FULL.csv
   - Residential (RES) TSI proportion view (ALL ROWS)
   - Columns: ID, POINT_X, POINT_Y, ORDER_ID_XY, storey bins (1, 2, 3, 5, 7, 10, 40), SUM

3. building_enrichment_tsi_proportions_{ISO3}_COM_FULL.csv
   - Commercial (COM) TSI proportion view (ALL ROWS)
   - Columns: ID, POINT_X, POINT_Y, ORDER_ID_XY, storey bins (1, 2, 3, 5, 7, 10, 40), SUM

4. building_enrichment_tsi_proportions_{ISO3}_IND_FULL.csv
   - Industrial (IND) TSI proportion view (ALL ROWS)
   - Columns: ID, POINT_X, POINT_Y, ORDER_ID_XY, storey bins (1, 2, 3, 5, 7, 10, 40), SUM

Storey Bin Mapping (for views):
-------------------------------
- 1:  Single storey
- 2:  Two storeys
- 3:  Three storeys
- 5:  4-5 storeys (mapped from "4_5")
- 7:  6-8 storeys (mapped from "6_8")
- 10: 9-20 storeys (mapped from "9_20")
- 40: 20+ storeys (mapped from "20")

Notes:
------
- This is a COMPLETE export of all data (not a sample)
- All percentages in views are TSI-weighted proportions
- SUM column in views represents the total across all storey bins for that grid cell
- Main output table contains both raw counts and TSI-adjusted values
- Grid IDs link records across all tables/views

File Sizes:
-----------
These are FULL dataset exports and may be large files (100+ MB each).
Estimated total size: 400-800 MB depending on data density.

Download Instructions:
---------------------
All files are saved in: {EXPORT_FOLDER}

To download from Databricks:
1. Navigate to the Volumes path in the Databricks UI
2. Select the files you want
3. Click "Download"

Or use dbutils in a notebook:
    dbutils.fs.cp("{EXPORT_FOLDER}/filename.csv", "file:/tmp/filename.csv")
    # Then download from /tmp/ using Databricks file browser

For questions or issues, refer to the project documentation.
"""

readme_path = f"{EXPORT_FOLDER}/README.txt"
try:
    # Write to temp first
    temp_readme = "/tmp/README_FULL.txt"
    with open(temp_readme, "w") as f:
        f.write(readme_content)
    
    # Copy to Volumes
    dbutils.fs.cp(f"file:{temp_readme}", readme_path)
    os.remove(temp_readme)
    
    print(f"✓ Created README.txt")
    exported_files.append(readme_path)
except Exception as e:
    print(f"⚠️  Warning: Could not create README: {e}")

# Calculate total size
print("\n" + "="*80)
print("EXPORT SUMMARY")
print("="*80)

total_size_mb = 0
total_size_gb = 0

for file_path in exported_files:
    try:
        file_info = dbutils.fs.ls(file_path)
        if file_info:
            size_bytes = file_info[0].size
            size_mb = size_bytes / (1024 * 1024)
            total_size_mb += size_mb
            
            # Display in MB or GB depending on size
            if size_mb >= 1024:
                size_gb = size_mb / 1024
                print(f"  ✓ {os.path.basename(file_path):60} ({size_gb:.2f} GB)")
            else:
                print(f"  ✓ {os.path.basename(file_path):60} ({size_mb:.2f} MB)")
        else:
            print(f"  ⚠️  {os.path.basename(file_path):60} (not found)")
    except Exception as e:
        print(f"  ⚠️  {os.path.basename(file_path):60} (error: {e})")

total_size_gb = total_size_mb / 1024

if total_size_mb >= 1024:
    print(f"\n{'Total size:':62} {total_size_gb:.2f} GB")
else:
    print(f"\n{'Total size:':62} {total_size_mb:.2f} MB")
    
print(f"{'Files exported:':62} {len(exported_files)}")

print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print(f"\n📁 Your FULL dataset files are ready in:\n   {EXPORT_FOLDER}\n")

print("Files available:")
for file_path in exported_files:
    print(f"  • {os.path.basename(file_path)}")

print("\n⚠️  NOTE: These are FULL dataset exports and may be large files (100+ MB each)")
print("    Total estimated size: 400-800 MB")

print("\n" + "="*80)
print("DOWNLOAD INSTRUCTIONS")
print("="*80)
print("""
Option 1 - Databricks UI (Recommended for large files):
  1. Navigate to: Data Explorer → Volumes → Browse to export folder
  2. Select files one at a time (large files may take time)
  3. Click Download

Option 2 - Copy to /tmp/ first:
  Run this in a notebook cell:
  
  for file in dbutils.fs.ls("{export_folder}"):
      if file.name.endswith(".csv") or file.name.endswith(".txt"):
          print(f"Copying {{file.name}}...")
          dbutils.fs.cp(file.path, f"file:/tmp/{{file.name}}")
          print(f"  ✓ Copied to /tmp/{{file.name}}")
  
  Then download from /tmp/ via File menu

Option 3 - Direct paths (for programmatic access):
""".format(export_folder=EXPORT_FOLDER))

for file_path in exported_files:
    print(f"  {file_path}")

print("\n" + "="*80)
print("⏱️  Note: Full dataset export may take 5-15 minutes depending on cluster size")
print("="*80)
# #!/usr/bin/env python3
# """
# Export Building Enrichment Samples - FIXED VERSION

# Creates clean CSV files (not folders) using Pandas for small samples.
# Saves directly to Volumes without ZIP to avoid filesystem issues.

# Usage:
#     python export_building_enrichment_samples_fixed.py --iso3 IND --sample_size 10000
# """

# import os
# import sys
# from datetime import datetime

# from pyspark.sql import SparkSession
# os.environ["CONFIG_PATH"] = "./config.json"
# # Parse command line arguments
# def parse_args():
#     args = {"iso3": "IND", "sample_size": 10000}
#     if len(sys.argv) > 1:
#         for i in range(1, len(sys.argv), 2):
#             if sys.argv[i].startswith("--"):
#                 key = sys.argv[i].lstrip("--")
#                 if i + 1 < len(sys.argv):
#                     value = sys.argv[i + 1]
#                     if key == "sample_size":
#                         args[key] = int(value)
#                     else:
#                         args[key] = value
#     return args

# args = parse_args()
# ISO3 = args["iso3"].upper()
# SAMPLE_SIZE = args["sample_size"]

# # Configuration
# CATALOG = "prp_mr_bdap_projects"
# SCHEMA = "geospatialsolutions"
# BASE_OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.building_enrichment_output_{ISO3.lower()}"
# BASE_VIEW_NAME = f"{CATALOG}.{SCHEMA}.building_enrichment_tsi_proportions_{ISO3.lower()}"

# # Export paths - save directly to Volumes as individual CSVs
# TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# EXPORT_FOLDER = f"/Volumes/{CATALOG}/{SCHEMA}/external/jrc/data/outputs/exports/{ISO3}_{SAMPLE_SIZE}_{TIMESTAMP}"

# # LOBs to export
# LOBS = ["res", "com", "ind"]

# print("="*80)
# print("BUILDING ENRICHMENT DATA EXPORT - FIXED VERSION")
# print("="*80)
# print(f"ISO3:          {ISO3}")
# print(f"Sample size:   {SAMPLE_SIZE:,} rows")
# print(f"Catalog:       {CATALOG}")
# print(f"Schema:        {SCHEMA}")
# print(f"Export folder: {EXPORT_FOLDER}")
# print(f"Output format: Individual CSV files (no ZIP)")
# print("="*80)

# # Initialize Spark
# spark = SparkSession.builder.getOrCreate()

# def export_table_to_csv(table_name, output_path, sample_size, description):
#     """
#     Export a sample to a single clean CSV file using Pandas.
    
#     Args:
#         table_name: Fully qualified table name
#         output_path: Full path including .csv filename (Volumes path)
#         sample_size: Number of rows to sample
#         description: Human-readable description for logging
    
#     Returns:
#         True if successful, False otherwise
#     """
#     print(f"\n{description}")
#     print(f"  Table: {table_name}")
#     print(f"  Output: {output_path}")
    
#     try:
#         # Check if table exists
#         if not spark.catalog.tableExists(table_name):
#             print(f"  ⚠️  WARNING: Table {table_name} does not exist. Skipping.")
#             return False
            
#         # Load and sample
#         df = spark.table(table_name)
#         row_count = df.count()
#         print(f"  Total rows in table: {row_count:,}")
        
#         if row_count == 0:
#             print(f"  ⚠️  WARNING: Table is empty. Skipping.")
#             return False
        
#         # Take sample and convert to Pandas
#         sample_df = df.limit(sample_size)
#         actual_sample = sample_df.count()
#         print(f"  Sampling {actual_sample:,} rows...")
        
#         # Convert to Pandas (efficient for 10k-50k rows)
#         print(f"  Converting to Pandas...")
#         pdf = sample_df.toPandas()
        
#         # Write to /tmp/ first (local filesystem), then copy to Volumes
#         temp_csv = f"/tmp/{os.path.basename(output_path)}"
        
#         print(f"  Writing CSV file to temp...")
#         pdf.to_csv(temp_csv, index=False)
        
#         # Copy from local /tmp/ to Volumes using dbutils
#         print(f"  Copying to Volumes...")
#         try:
#             from pyspark.dbutils import DBUtils
#             dbutils = DBUtils(spark)
#         except:
#             # Fallback for non-Databricks environments
#             import IPython
#             dbutils = IPython.get_ipython().user_ns.get('dbutils')
        
#         dbutils.fs.cp(f"file:{temp_csv}", output_path)
        
#         # Get file size
#         file_info = dbutils.fs.ls(output_path)
#         file_size_mb = file_info[0].size / (1024 * 1024) if file_info else 0
        
#         # Clean up temp file
#         os.remove(temp_csv)
        
#         print(f"  ✓ Export complete ({file_size_mb:.2f} MB)")
#         return True
        
#     except Exception as e:
#         print(f"  ❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # Create export folder using dbutils
# try:
#     from pyspark.dbutils import DBUtils
#     dbutils = DBUtils(spark)
# except:
#     # Fallback for notebooks where dbutils is already available
#     import IPython
#     dbutils = IPython.get_ipython().user_ns.get('dbutils')

# try:
#     # Try to create the folder
#     dbutils.fs.mkdirs(EXPORT_FOLDER)
#     print(f"\n✓ Created export folder: {EXPORT_FOLDER}")
# except Exception as e:
#     # Folder might already exist, which is fine
#     print(f"\n✓ Using export folder: {EXPORT_FOLDER}")

# print("\n" + "="*80)
# print("EXPORTING DATA")
# print("="*80)

# exported_files = []

# # 1. Export main output table
# main_csv = f"{EXPORT_FOLDER}/building_enrichment_output_{ISO3}_{SAMPLE_SIZE}.csv"
# if export_table_to_csv(
#     BASE_OUTPUT_TABLE, 
#     main_csv,
#     SAMPLE_SIZE,
#     f"[1/4] Exporting main output table ({SAMPLE_SIZE:,} rows)"
# ):
#     exported_files.append(main_csv)

# # 2-4. Export LOB views
# for idx, lob in enumerate(LOBS, start=2):
#     view_name = f"{BASE_VIEW_NAME}_{lob}_view"
#     view_csv = f"{EXPORT_FOLDER}/building_enrichment_tsi_proportions_{ISO3}_{lob.upper()}_{SAMPLE_SIZE}.csv"
    
#     if export_table_to_csv(
#         view_name,
#         view_csv,
#         SAMPLE_SIZE,
#         f"[{idx}/4] Exporting {lob.upper()} view ({SAMPLE_SIZE:,} rows)"
#     ):
#         exported_files.append(view_csv)

# # Create README file
# print("\n" + "="*80)
# print("CREATING README")
# print("="*80)

# readme_content = f"""Building Enrichment Data Export
# ================================

# Export Details:
# - ISO3 Code: {ISO3}
# - Sample Size: {SAMPLE_SIZE:,} rows per table/view
# - Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# - Catalog: {CATALOG}
# - Schema: {SCHEMA}

# Files Included:
# ---------------
# 1. building_enrichment_output_{ISO3}_{SAMPLE_SIZE}.csv
#    - Main building enrichment output table
#    - Contains all building counts, storey distributions, TSI values, and percentages
#    - Columns include: GRID_ID, lat, lon, urban, nr* columns, storey* columns, TSI columns, percentage columns

# 2. building_enrichment_tsi_proportions_{ISO3}_RES_{SAMPLE_SIZE}.csv
#    - Residential (RES) TSI proportion view
#    - Columns: ID, POINT_X, POINT_Y, ORDER_ID_XY, storey bins (1, 2, 3, 5, 7, 10, 40), SUM

# 3. building_enrichment_tsi_proportions_{ISO3}_COM_{SAMPLE_SIZE}.csv
#    - Commercial (COM) TSI proportion view
#    - Columns: ID, POINT_X, POINT_Y, ORDER_ID_XY, storey bins (1, 2, 3, 5, 7, 10, 40), SUM

# 4. building_enrichment_tsi_proportions_{ISO3}_IND_{SAMPLE_SIZE}.csv
#    - Industrial (IND) TSI proportion view
#    - Columns: ID, POINT_X, POINT_Y, ORDER_ID_XY, storey bins (1, 2, 3, 5, 7, 10, 40), SUM

# Storey Bin Mapping (for views):
# -------------------------------
# - 1:  Single storey
# - 2:  Two storeys
# - 3:  Three storeys
# - 5:  4-5 storeys (mapped from "4_5")
# - 7:  6-8 storeys (mapped from "6_8")
# - 10: 9-20 storeys (mapped from "9_20")
# - 40: 20+ storeys (mapped from "20")

# Notes:
# ------
# - All percentages in views are TSI-weighted proportions
# - SUM column in views represents the total across all storey bins for that grid cell
# - Main output table contains both raw counts and TSI-adjusted values
# - Grid IDs link records across all tables/views

# Download Instructions:
# ---------------------
# All files are saved in: {EXPORT_FOLDER}

# To download from Databricks:
# 1. Navigate to the Volumes path in the Databricks UI
# 2. Select the files you want
# 3. Click "Download"

# Or use dbutils in a notebook:
#     dbutils.fs.cp("{EXPORT_FOLDER}/filename.csv", "file:/tmp/filename.csv")
#     # Then download from /tmp/ using Databricks file browser

# For questions or issues, refer to the project documentation.
# """

# readme_path = f"{EXPORT_FOLDER}/README.txt"
# try:
#     # Write to temp first
#     temp_readme = "/tmp/README.txt"
#     with open(temp_readme, "w") as f:
#         f.write(readme_content)
    
#     # Copy to Volumes
#     dbutils.fs.cp(f"file:{temp_readme}", readme_path)
#     os.remove(temp_readme)
    
#     print(f"✓ Created README.txt")
#     exported_files.append(readme_path)
# except Exception as e:
#     print(f"⚠️  Warning: Could not create README: {e}")

# # Calculate total size
# print("\n" + "="*80)
# print("EXPORT SUMMARY")
# print("="*80)

# total_size_mb = 0
# for file_path in exported_files:
#     try:
#         file_info = dbutils.fs.ls(file_path)
#         if file_info:
#             size_mb = file_info[0].size / (1024 * 1024)
#             total_size_mb += size_mb
#             print(f"  ✓ {os.path.basename(file_path):60} ({size_mb:.2f} MB)")
#         else:
#             print(f"  ⚠️  {os.path.basename(file_path):60} (not found)")
#     except Exception as e:
#         print(f"  ⚠️  {os.path.basename(file_path):60} (error: {e})")

# print(f"\n{'Total size:':62} {total_size_mb:.2f} MB")
# print(f"{'Files exported:':62} {len(exported_files)}")

# print("\n" + "="*80)
# print("EXPORT COMPLETE")
# print("="*80)
# print(f"\n📁 Your files are ready in:\n   {EXPORT_FOLDER}\n")

# print("Files available:")
# for file_path in exported_files:
#     print(f"  • {os.path.basename(file_path)}")

# print("\n" + "="*80)
# print("DOWNLOAD INSTRUCTIONS")
# print("="*80)
# print("""
# Option 1 - Databricks UI:
#   1. Navigate to: Data Explorer → Volumes → Browse to export folder
#   2. Select files and click Download

# Option 2 - Notebook:
#   Run this in a cell to copy to /tmp/ for download:
  
#   for file in dbutils.fs.ls("{export_folder}"):
#       if file.name.endswith(".csv") or file.name.endswith(".txt"):
#           dbutils.fs.cp(file.path, f"file:/tmp/{{file.name}}")
#           print(f"Copied: {{file.name}}")
  
#   Then download from /tmp/ via File menu → Open → /tmp/

# Option 3 - Direct paths (for programmatic access):
# """.format(export_folder=EXPORT_FOLDER))

# for file_path in exported_files:
#     print(f"  {file_path}")

# print("\n" + "="*80)
