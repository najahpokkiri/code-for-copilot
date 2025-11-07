# os.environ["CONFIG_PATH"] = "./config.json"
#!/usr/bin/env python3

"""
Export Building Enrichment FULL Datasets + Excel Summary
Overwriting version (keeps only one export per country)
"""

import os
import sys
import shutil
import pandas as pd
from io import BytesIO
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import NullType
os.environ["CONFIG_PATH"] = "./config.json"
# ---------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
CATALOG = "prp_mr_bdap_projects"
SCHEMA = "geospatialsolutions"
BASE_OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.building_enrichment_output_{ISO3.lower()}"
BASE_VIEW_NAME = f"{CATALOG}.{SCHEMA}.building_enrichment_tsi_proportions_{ISO3.lower()}"

# Constant export folder per ISO3
EXPORT_FOLDER = f"/Volumes/{CATALOG}/{SCHEMA}/external/jrc/data/outputs/exports/FULL_{ISO3}"

LOBS = ["res", "com", "ind"]

# Toggle Excel summary
GENERATE_SUMMARY_EXCEL = True

# Excel summary config
INPUT_TABLE = f"{CATALOG}.{SCHEMA}.building_enrichment_output_{ISO3.lower()}"
OUTPUT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/external/jrc/data/outputs/exports/building_summary_country_layout_{ISO3}.xlsx"

# ---------------------------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------------------------
print("="*80)
print("BUILDING ENRICHMENT FULL DATASET EXPORT + EXCEL SUMMARY")
print("="*80)
print(f"ISO3:          {ISO3}")
print(f"Catalog:       {CATALOG}")
print(f"Schema:        {SCHEMA}")
print(f"Export folder: {EXPORT_FOLDER}")
print("="*80)

spark = SparkSession.builder.getOrCreate()

try:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
except:
    import IPython
    dbutils = IPython.get_ipython().user_ns.get('dbutils')

# ---------------------------------------------------------------------
# CLEAN PREVIOUS EXPORTS
# ---------------------------------------------------------------------
print(f"🧹 Cleaning old exports from {EXPORT_FOLDER}")
try:
    dbutils.fs.rm(EXPORT_FOLDER, recurse=True)
except Exception as e:
    print(f"Warning: could not remove old folder ({e})")
dbutils.fs.mkdirs(EXPORT_FOLDER)

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def clean_dataframe_for_csv(df):
    removed_columns = []
    print(f"  Checking {len(df.columns)} columns for CSV compatibility...")
    for field in df.schema.fields:
        if isinstance(field.dataType, NullType):
            removed_columns.append(field.name)
            continue
        dtype_str = str(field.dataType).upper()
        dtype_name = field.dataType.typeName().upper()
        if any(x in dtype_str for x in ["VOID", "NULL"]) or dtype_name in ["VOID", "NULL"]:
            removed_columns.append(field.name)
    if removed_columns:
        df = df.drop(*removed_columns)
    return df, removed_columns

def export_via_spark(df, temp_folder, output_path):
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_folder)
    csv_files = [f for f in os.listdir(temp_folder) if f.endswith(".csv") and not f.startswith("_")]
    csv_path = os.path.join(temp_folder, csv_files[0])
    dbutils.fs.cp(f"file:{csv_path}", output_path, True)
    size = os.path.getsize(csv_path) / (1024 * 1024)
    shutil.rmtree(temp_folder)
    return size

def export_via_pandas(df, output_path, max_rows=1000000):
    count = df.count()
    if count > max_rows:
        return None
    pdf = df.toPandas()
    temp_csv = f"/tmp/{os.path.basename(output_path)}"
    pdf.to_csv(temp_csv, index=False)
    dbutils.fs.cp(f"file:{temp_csv}", output_path, True)
    size = os.path.getsize(temp_csv) / (1024 * 1024)
    os.remove(temp_csv)
    return size

def export_full_table_to_csv(table_name, output_path, description):
    print(f"\n{description}")
    print(f"  Table: {table_name}")
    if not spark.catalog.tableExists(table_name):
        print(f"  ⚠️ Table does not exist")
        return False
    df = spark.table(table_name)
    count = df.count()
    if count == 0:
        print("  ⚠️ Table empty")
        return False
    df, removed = clean_dataframe_for_csv(df)
    temp_folder = f"/tmp/spark_export_{os.path.basename(output_path).replace('.csv', '')}"
    try:
        size = export_via_spark(df, temp_folder, output_path)
        print(f"  ✓ Exported via Spark ({size:.2f} MB, {count:,} rows)")
        return True
    except Exception as e:
        print(f"  Spark export failed: {e}")
        try:
            size = export_via_pandas(df, output_path)
            if size:
                print(f"  ✓ Exported via Pandas ({size:.2f} MB, {count:,} rows)")
                return True
        except Exception as e2:
            print(f"  Pandas export failed: {e2}")
    return False

# ---------------------------------------------------------------------
# EXPORT SECTION
# ---------------------------------------------------------------------
exported_files = []
export_results = []

print("\n" + "="*80)
print("EXPORTING FULL DATASETS")
print("="*80)

main_csv = f"{EXPORT_FOLDER}/building_enrichment_output_{ISO3}_FULL.csv"
success = export_full_table_to_csv(BASE_OUTPUT_TABLE, main_csv, "[1/4] Export main output")
export_results.append(("Main Output", success))
if success:
    exported_files.append(main_csv)

for idx, lob in enumerate(LOBS, start=2):
    view_name = f"{BASE_VIEW_NAME}_{lob}_view"
    view_csv = f"{EXPORT_FOLDER}/building_enrichment_tsi_proportions_{ISO3}_{lob.upper()}_FULL.csv"
    success = export_full_table_to_csv(view_name, view_csv, f"[{idx}/4] Export {lob.upper()} view")
    export_results.append((f"{lob.upper()} View", success))
    if success:
        exported_files.append(view_csv)

# ---------------------------------------------------------------------
# EXCEL SUMMARY GENERATION
# ---------------------------------------------------------------------
if GENERATE_SUMMARY_EXCEL:
    print("\n" + "="*80)
    print("GENERATING COUNTRY SUMMARY EXCEL")
    print("="*80)

    storeys = ["1", "2", "3", "4_5", "6_8", "9_20", "20", "SUM"]
    storey_cols = ["storey1", "storey2", "storey3", "storey4_5", "storey6_8", "storey9_20", "storey20"]
    building_types = ["RES", "COM", "IND"]
    area_map = {0: "RURAL", 1: "URBAN", 2: "SUBURBAN"}

    def col(s, t): return f"{s}_{t}"
    def tsi_col(s, t): return f"{s}_{t}_TSI"

    def summarize(df, btype):
        sums = [df[col(s, btype)].sum() for s in storey_cols]
        total = sum(sums)
        percs = [x / total * 100 if total else 0 for x in sums]
        sums.append(total)
        percs.append(100.0)
        tsi_sums = [df[tsi_col(s, btype)].sum() for s in storey_cols]
        tsi_total = sum(tsi_sums)
        tsi_percs = [x / tsi_total * 100 if tsi_total else 0 for x in tsi_sums]
        tsi_sums.append(tsi_total)
        tsi_percs.append(100.0)
        data = {
            "Number of Building Pixels": sums,
            "Percentage Building Pixels": percs,
            "Values Sum": tsi_sums,
            "Values Percentage": tsi_percs,
        }
        return pd.DataFrame(data, index=storeys)

    def build_area_tables(df, btype):
        results = {"OVERALL": summarize(df, btype)}
        df["area_label"] = df["urban"].map(area_map).fillna("RURAL")
        for label, subset in df.groupby("area_label"):
            results[label.upper()] = summarize(subset, btype)
        return results

    df_spark = spark.table(INPUT_TABLE)
    df = df_spark.toPandas()
    all_results = {b: build_area_tables(df, b) for b in building_types}

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        ws = workbook.add_worksheet("Country Summary")
        writer.sheets["Country Summary"] = ws

        fmt_title = workbook.add_format({"bold": True, "font_size": 14})
        fmt_header = workbook.add_format({"align": "center", "bold": True, "bg_color": "#FFD966", "border": 1})
        fmt_orange = workbook.add_format({"align": "center", "bold": True, "bg_color": "#F4B083", "border": 1})
        fmt_green = workbook.add_format({"align": "center", "bold": True, "bg_color": "#A9D18E", "border": 1})
        fmt_blue = workbook.add_format({"align": "center", "bold": True, "bg_color": "#9DC3E6", "border": 1})
        fmt_cell = workbook.add_format({"num_format": "#,##0", "border": 1})
        fmt_pct = workbook.add_format({"num_format": "0.00%", "border": 1})
        fmt_small = workbook.add_format({"font_size": 9, "border": 1})
        color_fmt = {"OVERALL": fmt_header, "URBAN": fmt_orange, "RURAL": fmt_green, "SUBURBAN": fmt_blue}

        row = 0
        for btype in building_types:
            ws.merge_range(row, 0, row, 60, f"{btype}IDENTIAL" if btype=="RES" else ("COMMERCIAL" if btype=="COM" else "INDUSTRIAL"), fmt_title)
            row += 1
            area_tables = all_results[btype]
            col_offset = 0
            for area, table in area_tables.items():
                ws.merge_range(row, col_offset, row, col_offset + len(storeys), area, color_fmt.get(area, fmt_header))
                ws.write(row + 1, col_offset, "STOREYS", fmt_small)
                for i, s in enumerate(storeys):
                    ws.write(row + 1, col_offset + i + 1, s, fmt_small)
                for ridx, metric_name in enumerate(table.columns):
                    ws.write(row + 2 + ridx, col_offset, metric_name, fmt_small)
                    for cidx, val in enumerate(table.iloc[:, ridx]):
                        fmt = fmt_cell if "Percentage" not in metric_name else fmt_pct
                        ws.write(row + 2 + ridx, col_offset + cidx + 1, val/100 if "Percentage" in metric_name else val, fmt)
                col_offset += len(storeys) + 3
            row += 8
        ws.set_column(0, 200, 14)

    with open("/tmp/building_summary_country_layout.xlsx", "wb") as f:
        f.write(output.getbuffer())
    dbutils.fs.cp("file:/tmp/building_summary_country_layout.xlsx", OUTPUT_PATH, True)
    os.remove("/tmp/building_summary_country_layout.xlsx")
    print(f"✓ Excel summary written to {OUTPUT_PATH}")
    exported_files.append(OUTPUT_PATH)

# ---------------------------------------------------------------------
# README CREATION
# ---------------------------------------------------------------------
readme_content = f"""Building Enrichment FULL Dataset Export
=========================================

Export Details:
- ISO3 Code: {ISO3}
- Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Method: Robust export with VOID column handling
- Overwrite mode: ON (keeps one export per country)

Export Status:
--------------
"""
for name, success in export_results:
    status = "✓ Success" if success else "✗ Failed"
    readme_content += f"- {name}: {status}\n"

readme_path = f"{EXPORT_FOLDER}/README.txt"
temp_readme = "/tmp/README_FULL.txt"
with open(temp_readme, "w") as f:
    f.write(readme_content)
dbutils.fs.cp(f"file:{temp_readme}", readme_path, True)
os.remove(temp_readme)
exported_files.append(readme_path)
print("✓ README created")

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
for fpath in exported_files:
    print("  ✓", fpath)
print("\nAll exports complete, previous versions replaced.")

# #!/usr/bin/env python3
# """
# Export Building Enrichment FULL Datasets - MOST ROBUST VERSION

# This version:
# 1. Uses the most thorough VOID column detection
# 2. Falls back to Pandas for small datasets if Spark CSV fails
# 3. Has detailed logging for debugging

# Usage:
#     python export_full_dataset_robust.py --iso3 IND
# """

# import os
# import sys
# import shutil
# from datetime import datetime

# from pyspark.sql import SparkSession
# from pyspark.sql.types import NullType
# # os.environ["CONFIG_PATH"] = "./config.json"
# # Parse command line arguments
# def parse_args():
#     args = {"iso3": "IND"}
#     if len(sys.argv) > 1:
#         for i in range(1, len(sys.argv), 2):
#             if sys.argv[i].startswith("--"):
#                 key = sys.argv[i].lstrip("--")
#                 if i + 1 < len(sys.argv):
#                     args[key] = sys.argv[i + 1]
#     return args

# args = parse_args()
# ISO3 = args["iso3"].upper()

# # Configuration
# CATALOG = "prp_mr_bdap_projects"
# SCHEMA = "geospatialsolutions"
# BASE_OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.building_enrichment_output_{ISO3.lower()}"
# BASE_VIEW_NAME = f"{CATALOG}.{SCHEMA}.building_enrichment_tsi_proportions_{ISO3.lower()}"

# # Export paths
# TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# EXPORT_FOLDER = f"/Volumes/{CATALOG}/{SCHEMA}/external/jrc/data/outputs/exports/FULL_{ISO3}_{TIMESTAMP}"

# # LOBs to export
# LOBS = ["res", "com", "ind"]

# print("="*80)
# print("BUILDING ENRICHMENT FULL DATASET EXPORT - ROBUST VERSION")
# print("="*80)
# print(f"ISO3:          {ISO3}")
# print(f"Export size:   FULL DATASET (ALL ROWS)")
# print(f"Catalog:       {CATALOG}")
# print(f"Schema:        {SCHEMA}")
# print(f"Export folder: {EXPORT_FOLDER}")
# print("="*80)

# # Initialize Spark
# spark = SparkSession.builder.getOrCreate()

# # Get dbutils
# try:
#     from pyspark.dbutils import DBUtils
#     dbutils = DBUtils(spark)
# except:
#     import IPython
#     dbutils = IPython.get_ipython().user_ns.get('dbutils')

# def clean_dataframe_for_csv(df):
#     """
#     Remove all columns that cannot be written to CSV.
#     Returns cleaned dataframe and list of removed columns.
#     """
#     removed_columns = []
    
#     print(f"  Checking {len(df.columns)} columns for CSV compatibility...")
    
#     for field in df.schema.fields:
#         # Check if it's a NullType (the actual type for VOID columns)
#         if isinstance(field.dataType, NullType):
#             removed_columns.append(field.name)
#             print(f"    Removing NullType column: {field.name}")
#             continue
        
#         # Check string representations
#         dtype_str = str(field.dataType).upper()
#         dtype_name = field.dataType.typeName().upper()
        
#         if any(x in dtype_str for x in ["VOID", "NULL"]):
#             removed_columns.append(field.name)
#             print(f"    Removing column with VOID/NULL type: {field.name} ({field.dataType})")
#             continue
            
#         if dtype_name in ["VOID", "NULL"]:
#             removed_columns.append(field.name)
#             print(f"    Removing column with type name {dtype_name}: {field.name}")
#             continue
    
#     if removed_columns:
#         print(f"  Dropping {len(removed_columns)} incompatible columns")
#         df = df.drop(*removed_columns)
#     else:
#         print(f"  All columns are CSV-compatible")
    
#     return df, removed_columns

# def export_via_spark(df, temp_folder, output_path):
#     """Export using Spark CSV writer"""
#     print(f"  Method: Spark CSV writer")
    
#     # Remove temp folder if exists
#     if os.path.exists(temp_folder):
#         shutil.rmtree(temp_folder)
#     os.makedirs(temp_folder, exist_ok=True)
    
#     # Write with Spark
#     df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_folder)
    
#     # Find CSV file
#     csv_files = [f for f in os.listdir(temp_folder) if f.endswith(".csv") and not f.startswith("_")]
#     if not csv_files:
#         raise Exception("No CSV file generated by Spark")
    
#     csv_path = os.path.join(temp_folder, csv_files[0])
#     file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    
#     # Copy to Volumes
#     dbutils.fs.cp(f"file:{csv_path}", output_path)
    
#     # Cleanup
#     shutil.rmtree(temp_folder)
    
#     return file_size_mb

# def export_via_pandas(df, output_path, max_rows=1000000):
#     """Export using Pandas (for smaller datasets or as fallback)"""
#     print(f"  Method: Pandas (fallback)")
    
#     row_count = df.count()
#     if row_count > max_rows:
#         print(f"  ⚠️  Dataset too large for Pandas export ({row_count:,} > {max_rows:,})")
#         return None
    
#     # Convert to Pandas
#     print(f"  Converting {row_count:,} rows to Pandas...")
#     pdf = df.toPandas()
    
#     # Write to temp file
#     temp_csv = f"/tmp/{os.path.basename(output_path)}"
#     print(f"  Writing CSV to temp...")
#     pdf.to_csv(temp_csv, index=False)
    
#     file_size_mb = os.path.getsize(temp_csv) / (1024 * 1024)
    
#     # Copy to Volumes
#     print(f"  Copying to Volumes...")
#     dbutils.fs.cp(f"file:{temp_csv}", output_path)
    
#     # Cleanup
#     os.remove(temp_csv)
    
#     return file_size_mb

# def export_full_table_to_csv(table_name, output_path, description):
#     """
#     Export FULL table to CSV with robust error handling.
#     """
#     print(f"\n{description}")
#     print(f"  Table: {table_name}")
#     print(f"  Output: {output_path}")
    
#     temp_folder = None
    
#     try:
#         # Check if table exists
#         if not spark.catalog.tableExists(table_name):
#             print(f"  ⚠️  Table does not exist")
#             return False
        
#         # Load table
#         df = spark.table(table_name)
#         row_count = df.count()
        
#         print(f"  Total rows: {row_count:,}")
#         print(f"  Original columns: {len(df.columns)}")
        
#         if row_count == 0:
#             print(f"  ⚠️  Table is empty")
#             return False
        
#         # Clean dataframe
#         df, removed_cols = clean_dataframe_for_csv(df)
#         print(f"  Columns after cleaning: {len(df.columns)}")
        
#         if removed_cols:
#             print(f"  Removed columns: {', '.join(removed_cols)}")
        
#         # Try Spark export first
#         temp_folder = f"/tmp/spark_export_{os.path.basename(output_path).replace('.csv', '')}"
        
#         try:
#             file_size_mb = export_via_spark(df, temp_folder, output_path)
#             print(f"  ✓ Export complete via Spark ({file_size_mb:.2f} MB, {row_count:,} rows)")
#             return True
            
#         except Exception as spark_error:
#             print(f"  ⚠️  Spark export failed: {spark_error}")
#             print(f"  Trying Pandas fallback...")
            
#             # Try Pandas as fallback
#             try:
#                 file_size_mb = export_via_pandas(df, output_path)
#                 if file_size_mb:
#                     print(f"  ✓ Export complete via Pandas ({file_size_mb:.2f} MB, {row_count:,} rows)")
#                     return True
#                 else:
#                     print(f"  ❌ Dataset too large for Pandas fallback")
#                     return False
                    
#             except Exception as pandas_error:
#                 print(f"  ❌ Pandas export also failed: {pandas_error}")
#                 return False
        
#     except Exception as e:
#         print(f"  ❌ ERROR: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
        
#     finally:
#         # Cleanup
#         if temp_folder and os.path.exists(temp_folder):
#             try:
#                 shutil.rmtree(temp_folder)
#             except:
#                 pass

# # Create export folder
# try:
#     dbutils.fs.mkdirs(EXPORT_FOLDER)
#     print(f"\n✓ Created export folder: {EXPORT_FOLDER}")
# except:
#     print(f"\n✓ Using export folder: {EXPORT_FOLDER}")

# print("\n" + "="*80)
# print("EXPORTING FULL DATASETS")
# print("="*80)

# exported_files = []
# export_results = []

# # Export main output table
# print("\n" + "-"*80)
# main_csv = f"{EXPORT_FOLDER}/building_enrichment_output_{ISO3}_FULL.csv"
# success = export_full_table_to_csv(
#     BASE_OUTPUT_TABLE,
#     main_csv,
#     "[1/4] Exporting main output table"
# )
# export_results.append(("Main Output", success))
# if success:
#     exported_files.append(main_csv)

# # Export LOB views
# for idx, lob in enumerate(LOBS, start=2):
#     print("\n" + "-"*80)
#     view_name = f"{BASE_VIEW_NAME}_{lob}_view"
#     view_csv = f"{EXPORT_FOLDER}/building_enrichment_tsi_proportions_{ISO3}_{lob.upper()}_FULL.csv"
    
#     success = export_full_table_to_csv(
#         view_name,
#         view_csv,
#         f"[{idx}/4] Exporting {lob.upper()} view"
#     )
#     export_results.append((f"{lob.upper()} View", success))
#     if success:
#         exported_files.append(view_csv)

# # Create README
# print("\n" + "="*80)
# print("CREATING README")
# print("="*80)

# readme_content = f"""Building Enrichment FULL Dataset Export
# =========================================

# Export Details:
# - ISO3 Code: {ISO3}
# - Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# - Method: Robust export with VOID column handling

# Export Status:
# --------------
# """

# for name, success in export_results:
#     status = "✓ Success" if success else "✗ Failed"
#     readme_content += f"- {name}: {status}\n"

# readme_content += """

# Note: VOID/NULL type columns have been automatically removed for CSV compatibility.

# For full documentation, see project README.
# """

# readme_path = f"{EXPORT_FOLDER}/README.txt"
# try:
#     temp_readme = "/tmp/README_FULL.txt"
#     with open(temp_readme, "w") as f:
#         f.write(readme_content)
#     dbutils.fs.cp(f"file:{temp_readme}", readme_path)
#     os.remove(temp_readme)
#     print(f"✓ Created README.txt")
#     exported_files.append(readme_path)
# except Exception as e:
#     print(f"⚠️  Could not create README: {e}")

# # Summary
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
#             if size_mb >= 1024:
#                 print(f"  ✓ {os.path.basename(file_path):60} ({size_mb/1024:.2f} GB)")
#             else:
#                 print(f"  ✓ {os.path.basename(file_path):60} ({size_mb:.2f} MB)")
#     except Exception as e:
#         print(f"  ⚠️  {os.path.basename(file_path):60} (error: {e})")

# print(f"\n{'Total size:':62} {total_size_mb:.2f} MB")
# print(f"{'Files exported:':62} {len(exported_files)}")

# successful = sum(1 for _, s in export_results if s)
# print(f"{'Successful exports:':62} {successful}/{len(export_results)}")

# if successful < len(export_results):
#     print("\n⚠️  Some exports failed:")
#     for name, success in export_results:
#         if not success:
#             print(f"  ✗ {name}")

# print("\n" + "="*80)
# print("EXPORT COMPLETE")
# print("="*80)
# print(f"\n📁 Files ready in: {EXPORT_FOLDER}")