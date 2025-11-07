

#!/usr/bin/env python3
"""
Task 6 â€" Create per-LOB TSI proportion views (CSV-COMPATIBLE VERSION)

CRITICAL FIX: Uses CAST(NULL AS STRING) instead of NULL for missing columns.
This prevents VOID type columns that cannot be exported to CSV.

- ISO3 is the only dynamic value, read from config or env (default "IND").
- All table/view names are auto-computed with the ISO3 suffix.
- Creates views with proper data types for CSV export compatibility.
"""

import os
import sys
import json
import traceback
import re
from typing import Dict, List, Optional

from pyspark.sql import SparkSession
os.environ["CONFIG_PATH"] = "./config.json"
# Static settings
BASE_INPUT_TABLE = "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_output"
BASE_OUTPUT_VIEW = "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_tsi_proportions"
LOBS = ["RES", "COM", "IND"]
LEVEL_MAPPING = {
    "1": "1",
    "2": "2",
    "3": "3",
    "4_5": "5",
    "6_8": "7",
    "9_20": "10",
    "20": "40"
}
X_COL_CANDIDATES = ["centroid_x","lon","POINT_X","longitude"]
Y_COL_CANDIDATES = ["centroid_y","lat","POINT_Y","latitude"]
ORDER_COL_CANDIDATES = ["order_id","ORDER_ID_XY","ID_ORDER_XY","orderid"]
DROP_TABLE_IF_EXISTS = False
FALLBACK_VIEW_SUFFIX = "_view"

def get_iso3():
    iso3 = None
    cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG")
    if cfg_path:
        if cfg_path.startswith("dbfs:"):
            cfg_path = cfg_path.replace("dbfs:", "/dbfs", 1)
        try:
            with open(cfg_path, "r", encoding="utf8") as fh:
                j = json.load(fh)
            iso3 = j.get("iso3")
        except Exception:
            pass
    if not iso3:
        iso3 = os.environ.get("ISO3", "IND")
    return iso3.strip().lower()

def add_iso_suffix(name, iso3):
    if name is None:
        return name
    if name.lower().endswith(f"_{iso3}"):
        return name
    if name.endswith(".csv"):
        return name[:-4] + f"_{iso3}.csv"
    if name.endswith(".parquet"):
        return name[:-8] + f"_{iso3}.parquet"
    return f"{name}_{iso3}"

def find_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    upcols = [c.upper() for c in cols]
    for cand in candidates:
        for idx, uc in enumerate(upcols):
            if uc == cand.upper():
                return cols[idx]
    for cand in candidates:
        low = cand.lower()
        for idx, c in enumerate(cols):
            if low in c.lower():
                return cols[idx]
    return None

def create_view_for_lob(spark, input_table, output_view_base, cols, lob, iso3):
    out_base = output_view_base
    x_cands = X_COL_CANDIDATES
    y_cands = Y_COL_CANDIDATES
    order_cands = ORDER_COL_CANDIDATES
    drop_if_table = DROP_TABLE_IF_EXISTS
    fallback_suffix = FALLBACK_VIEW_SUFFIX
    level_map = LEVEL_MAPPING

    # Collect coordinate/order columns
    actual_x = find_first(cols, x_cands)
    actual_y = find_first(cols, y_cands)
    actual_order = find_first(cols, order_cands)
    actual_id = find_first(cols, ["GRID_ID","ID","grid_id","id"]) or "GRID_ID"

    # Find and map bin columns
    lob_cols = [c for c in cols if c.upper().startswith(lob.upper()+"_") and c.lower().endswith("_tsi_perc")]
    bin_map = {}
    for c in lob_cols:
        m = re.match(rf"{lob}_(Storey_)?(.+)_tsi_perc", c, re.IGNORECASE)
        if m:
            orig_level = m.group(2)
            mapped_bin = level_map.get(orig_level, orig_level)
            bin_map[orig_level] = (mapped_bin, c)
    
    # Only one column per mapped_bin (first occurrence)
    mapped_cols = {}
    for orig, (mapped_bin, c) in bin_map.items():
        if mapped_bin not in mapped_cols:
            mapped_cols[mapped_bin] = c

    # Sorted order as per mapping values
    bin_order = [v for k,v in level_map.items()]
    select_parts = []
    
    # CRITICAL FIX: Use CAST(NULL AS STRING) instead of just NULL
    # This prevents VOID type columns that can't be exported to CSV
    
    if actual_id in cols:
        select_parts.append(f"`{actual_id}` AS ID")
    else:
        select_parts.append(f"CAST(NULL AS STRING) AS ID")
    
    if actual_x:
        select_parts.append(f"`{actual_x}` AS POINT_X")
    else:
        select_parts.append(f"CAST(NULL AS DOUBLE) AS POINT_X")
    
    if actual_y:
        select_parts.append(f"`{actual_y}` AS POINT_Y")
    else:
        select_parts.append(f"CAST(NULL AS DOUBLE) AS POINT_Y")
    
    # ORDER_ID_XY: This was causing the VOID type error
    if actual_order:
        select_parts.append(f"`{actual_order}` AS ORDER_ID_XY")
    else:
        # IMPORTANT: Cast NULL to LONG (integer type) instead of leaving as VOID
        select_parts.append(f"CAST(NULL AS LONG) AS ORDER_ID_XY")

    for b in bin_order:
        c = mapped_cols.get(b)
        if c:
            select_parts.append(f"COALESCE(`{c}`, 0.0) AS `{b}`")
        else:
            select_parts.append(f"0.0 AS `{b}`")

    # SUM = sum of mapped bin columns in order
    sum_expr = " + ".join([f"`{b}`" for b in bin_order])
    select_parts.append(f"({sum_expr}) AS SUM")

    select_sql = ",\n  ".join(select_parts)
    short_view = f"{out_base}_{lob.lower()}"
    fq_view = short_view

    # Check for table collision (fallback if needed)
    try:
        schemaPrefix = ".".join(fq_view.split(".")[:-1])
        short_name = fq_view.split(".")[-1].lower()
        existing_tables = [r["tableName"].lower() for r in spark.sql(f"SHOW TABLES IN {schemaPrefix}").collect()] if "." in schemaPrefix else []
    except Exception:
        existing_tables = []
    if short_name in existing_tables and not drop_if_table:
        fq_view = fq_view + fallback_suffix

    create_sql = f"CREATE OR REPLACE VIEW {fq_view} AS\nSELECT\n  {select_sql}\nFROM {input_table}"
    
    print(f"\nCreating CSV-compatible view: {fq_view}")
    print("="*80)
    print(create_sql)
    print("="*80)
    
    spark.sql(create_sql)
    print(f"✅ Created view: {fq_view}")
    
    # Verify the view doesn't have VOID columns
    view_df = spark.table(fq_view)
    from pyspark.sql.types import NullType
    
    void_cols = []
    for field in view_df.schema.fields:
        if isinstance(field.dataType, NullType):
            void_cols.append(field.name)
    
    if void_cols:
        print(f"⚠️  WARNING: View still has VOID columns: {void_cols}")
        print("   This should not happen with the fixed SQL!")
    else:
        print(f"✅ View is CSV-compatible (no VOID columns)")

def main():
    try:
        iso3 = get_iso3()
        input_table = add_iso_suffix(BASE_INPUT_TABLE, iso3)
        output_view_base = add_iso_suffix(BASE_OUTPUT_VIEW, iso3)
        
        print("="*80)
        print("TASK 6: Create CSV-Compatible TSI Proportion Views")
        print("="*80)
        print(f"ISO3: {iso3}")
        print(f"Input table: {input_table}")
        print(f"Output view base: {output_view_base}")
        print("="*80)
        print()
        print("CRITICAL FIX APPLIED:")
        print("  - Using CAST(NULL AS <type>) instead of NULL")
        print("  - Prevents VOID type columns")
        print("  - Views will be CSV-exportable")
        print("="*80)
        
        spark = SparkSession.builder.getOrCreate()
        cols = spark.table(input_table).columns
        
        print(f"\nFound {len(cols)} columns in input table")
        print(f"\nCreating views for LOBs: {LOBS}")
        
        for lob in LOBS:
            create_view_for_lob(spark, input_table, output_view_base, cols, lob, iso3)
        
        print("\n" + "="*80)
        print("VIEW CREATION COMPLETE")
        print("="*80)
        print(f"✅ All {len(LOBS)} views created successfully")
        print("✅ Views are CSV-compatible and ready for export")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR OCCURRED")
        print("="*80)
        print(f"❌ {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()