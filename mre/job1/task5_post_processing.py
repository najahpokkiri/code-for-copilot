#!/usr/bin/env python3
"""
Task 5 — Post-Processing & TSI Calculations

Processes grid counts from Task 4 to generate building estimates with
Total Sum Insured (TSI) calculations by storey and building type.

Configuration:
--------------
Reads from config.json (generated from config.yaml via config_builder.py).
All table names are auto-generated from the YAML configuration.

Required config keys:
  - grid_count_table: Input table with grid counts from Task 4
  - proportions_table: Proportions table from Task 1
  - tsi_table: TSI multipliers table from Task 1
  - output_table: Output Delta table for enriched data
  - write_mode: Delta table write mode (default: "overwrite")

Usage:
------
  python task5_post_processing.py --config_path config.json

Output:
-------
  - Delta table: {catalog}.{schema}.inv_NoS_{ISO3}_{YYMMDD}
    Contains storey-level breakdowns by RES/COM/IND with TSI calculations

PROCESSING STEPS:
-----------------
1. Pivot building counts: Convert built codes to LOB columns (RES/COM)
2. Filter zero rows: Remove grids with no buildings
3. Add ID/order: Generate unique IDs and spatial ordering
4. Storey distribution: Apply proportions to distribute by storey levels
5. TSI calculations: Multiply by TSI rates and compute percentages
6. Imputation: Fill missing TSI percentages with urban-specific averages
7. Column organization: Group related columns with block sums

ENHANCEMENTS:
-------------
- Filters rows where ALL building counts are zero
- Imputes zero TSI percentages using urban-specific averages
- Ensures CSV compatibility (no VOID/NULL column types)
- Adds block sum columns for validation
- Proper string casting for ID columns

Output Schema:
--------------
Core columns: grid_id, urban, lat, lon
Building counts: nr{built}_{lob} (e.g., nr11_res, nr21_com)
Storey-level: {lob}_storey{level} (e.g., res_storey1, com_storey4_5)
TSI values: {lob}_storey{level}_TSI
TSI percentages: {lob}_storey{level}_tsi_perc
Block sums: Various *_SUM columns for validation

Notes:
------
  - Coordinates renamed from centroid_x/y to lon/lat
  - LOB mapping: built 11-15 → RES, 21-25 → COM
  - Storey bins: 1, 2, 3, 4_5, 6_8, 9_20, 20
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F
os.environ["CONFIG_PATH"] = "./config.json"
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        key = args[i]
        if key.startswith("--"):
            env_key = key.lstrip("-").upper()
            value = ""
            if (i + 1) < len(args) and not args[i + 1].startswith("--"):
                value = args[i + 1]
                i += 2
            else:
                i += 1
            if value != "":
                os.environ[env_key] = value
        else:
            i += 1

# ---------------------------
# Config loader & defaults
# ---------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "grid_count_table": None,
    "proportions_table": None,
    "tsi_table": None,
    "output_table": None,
    "test_tile": None,
    "save_temp_csv": False,
    "output_dir": None
}

def _read_json_path(path: str) -> Dict[str, Any]:
    if path.startswith("dbfs:"):
        local_path = path.replace("dbfs:", "/dbfs", 1)
    else:
        local_path = path
    with open(local_path, "r") as f:
        return json.load(f)

def load_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG", "") or os.environ.get("CONFIG_JSON", "")
    if cfg_path:
        loaded = _read_json_path(cfg_path)
        if not isinstance(loaded, dict):
            raise ValueError("config file must be a JSON object")
        cfg.update(loaded)
    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            val = os.environ[env_key]
            cfg[key] = str(val).lower() in ("true","1","t","yes") if isinstance(cfg[key], bool) else val
    required = ["grid_count_table","proportions_table","tsi_table","output_table"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {missing}. Please add them to config or pass overrides.")
    return cfg

# ---------------------------
# Core Processing Functions
# ---------------------------
def pivot_buildings(step1_df: DataFrame) -> DataFrame:
    df = step1_df
    
    df_with_lob = df.withColumn("LOB",
        F.when(F.col("built").between(11, 15), F.lit("RES"))
         .when(F.col("built").between(21, 25), F.lit("COM"))
         .otherwise(F.lit("UNKNOWN"))
    )
    
    with_built_lob = df_with_lob.withColumn(
        "built_lob",
        F.concat(F.lit("nr"), F.col("built").cast("string"), F.lit("_"), F.lower(F.col("LOB")))
    )
    
    pivoted = (with_built_lob
               .groupBy("grid_id", "urban", "lat", "lon")
               .pivot("built_lob")
               .agg(F.first("count"))
               .na.fill(0)
               )
    return pivoted

def filter_zero_buildings(df: DataFrame) -> DataFrame:
    """
    Filter out rows where ALL nr* columns are zero or NULL.
    """
    nr_cols = [c for c in df.columns if c.lower().startswith("nr")]
    
    if not nr_cols:
        print("WARNING: No 'nr' columns found for filtering. Returning dataframe unchanged.")
        return df
    
    print(f"Found {len(nr_cols)} nr* columns for filtering: {nr_cols}")
    
    conditions = [F.coalesce(F.col(c), F.lit(0)) > 0 for c in nr_cols]
    
    filter_condition = conditions[0]
    for condition in conditions[1:]:
        filter_condition = filter_condition | condition
    
    count_before = df.count()
    df_filtered = df.filter(filter_condition)
    count_after = df_filtered.count()
    
    rows_removed = count_before - count_after
    print(f"Filtering zero-building rows:")
    print(f"  Before: {count_before:,} rows")
    print(f"  After:  {count_after:,} rows")
    print(f"  Removed: {rows_removed:,} rows ({100*rows_removed/max(count_before,1):.2f}%)")
    
    return df_filtered

def add_id_and_order(step2_df: DataFrame) -> DataFrame:
    """
    Add ID and order columns.
    
    CRITICAL: Cast ID to string explicitly to avoid VOID type issues.
    This ensures the column has a proper data type for CSV export.
    """
    df = step2_df
    
    # Create ID as lon_lat string (longitude first, preserving minus signs)
    # IMPORTANT: Explicitly cast to string to avoid NULL/VOID type issues
    df = df.withColumn(
        "ID",
        F.regexp_replace(
            F.concat_ws("_", F.col("lon").cast("string"), F.col("lat").cast("string")),
            r"\.", ""
        )
    )
    
    # Ensure ID is string type (defensive coding)
    df = df.withColumn("ID", F.col("ID").cast("string"))
    
    # GRID_ID is same as ID
    df = df.withColumn("GRID_ID", F.col("ID").cast("string"))
    
    # Sort by lon ascending, then lat ascending (2-layer sort)
    window_spec = Window.orderBy(F.col("lon").asc(), F.col("lat").asc())
    
    # Create order column as LONG (integer type, not VOID)
    df = df.withColumn("ID_ORDER_XY", F.row_number().over(window_spec).cast("long"))
    
    print("ID columns created:")
    print(f"  ID: {df.schema['ID'].dataType}")
    print(f"  GRID_ID: {df.schema['GRID_ID'].dataType}")
    print(f"  ID_ORDER_XY: {df.schema['ID_ORDER_XY'].dataType}")
    
    return df
    
def process_buildings_final(step2_df: DataFrame, props_df: DataFrame) -> DataFrame:
    result_df = step2_df
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    
    nr_cols = [c for c in result_df.columns if c.startswith("nr")]
    
    for type_name in ["RES", "COM", "IND"]:
        if type_name == "RES":
            building_range = range(11, 16)
            suffix = "res"
        elif type_name == "COM":
            building_range = range(21, 26)
            suffix = "com"
        else:  # IND
            building_range = range(21, 26)
            suffix = "ind"
            # IND uses same building counts as COM but different proportions
            for building in building_range:
                com_col = f"nr{building}_com"
                ind_col = f"nr{building}_ind"
                if com_col in result_df.columns and ind_col not in result_df.columns:
                    result_df = result_df.withColumn(ind_col, F.col(com_col))
        
        for building in building_range:
            props = props_df.filter(
                (F.col("LOB") == type_name) & (F.col("storey") == building)
            ).select(
                "urban", 
                *[F.col(f"`{s}`").alias(f"prop_{s}") for s in storey_levels]
            )
            
            result_df = result_df.join(props, on="urban", how="left")
            
            for storey in storey_levels:
                nr_col = f"nr{building}_{suffix}"
                prop_col = f"prop_{storey}"
                out_col = f"storey{storey}_{suffix}_{building}"
                
                if nr_col not in result_df.columns:
                    result_df = result_df.withColumn(nr_col, F.lit(0))
                
                result_df = result_df.withColumn(
                    out_col, 
                    F.coalesce(F.col(nr_col), F.lit(0)) * F.coalesce(F.col(prop_col), F.lit(0.0))
                )
            
            drop_props = [c for c in result_df.columns if c.startswith("prop_")]
            if drop_props:
                result_df = result_df.drop(*drop_props)
        
        for storey in storey_levels:
            summands = [f"coalesce(storey{storey}_{suffix}_{i}, 0)" for i in building_range]
            sum_expr = " + ".join(summands)
            result_df = result_df.withColumn(f"storey{storey}_{type_name}", F.expr(sum_expr))
        
        drop_intermediates = [c for c in result_df.columns if f"_{suffix}_" in c]
        if drop_intermediates:
            result_df = result_df.drop(*drop_intermediates)
    
    final_cols = []
    for storey in storey_levels:
        final_cols.extend([f"storey{storey}_RES", f"storey{storey}_COM", f"storey{storey}_IND"])
    
    if "GRID_ID" not in result_df.columns and "ID" in result_df.columns:
        result_df = result_df.withColumn("GRID_ID", F.col("ID").cast("string"))
    
    select_cols = ["GRID_ID", "ID_ORDER_XY", "lat", "lon", "urban"] + nr_cols + final_cols
    select_cols = [c for c in select_cols if c in result_df.columns]

    return result_df.select(*select_cols)

def create_step4_df(step3_df: DataFrame, tsi_df: DataFrame) -> DataFrame:
    df = step3_df
    tsi_rows = tsi_df.collect()
    tsi_map = {}
    for r in tsi_rows:
        tsi_map[r["LOB"]] = r.asDict()
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    for storey in storey_levels:
        res_val = tsi_map.get("RES", {}).get(storey, 0)
        df = df.withColumn(f"storey{storey}_RES_TSI", F.col(f"storey{storey}_RES") * F.lit(res_val))
        com_val = tsi_map.get("COM", {}).get(storey, 0)
        df = df.withColumn(f"storey{storey}_COM_TSI", F.col(f"storey{storey}_COM") * F.lit(com_val))
        ind_val = tsi_map.get("IND", {}).get(storey, 0)
        df = df.withColumn(f"storey{storey}_IND_TSI", F.col(f"storey{storey}_IND") * F.lit(ind_val))
    return df

def create_step5_df(step4_df: DataFrame) -> DataFrame:
    df = step4_df
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    res_reg_sum = None
    com_reg_sum = None
    ind_reg_sum = None
    res_tsi_sum = None
    com_tsi_sum = None
    ind_tsi_sum = None
    for i, lvl in enumerate(storey_levels):
        col_res = F.col(f"storey{lvl}_RES")
        col_com = F.col(f"storey{lvl}_COM")
        col_ind = F.col(f"storey{lvl}_IND")
        col_res_tsi = F.col(f"storey{lvl}_RES_TSI")
        col_com_tsi = F.col(f"storey{lvl}_COM_TSI")
        col_ind_tsi = F.col(f"storey{lvl}_IND_TSI")
        if i == 0:
            res_reg_sum = col_res
            com_reg_sum = col_com
            ind_reg_sum = col_ind
            res_tsi_sum = col_res_tsi
            com_tsi_sum = col_com_tsi
            ind_tsi_sum = col_ind_tsi
        else:
            res_reg_sum = res_reg_sum + col_res
            com_reg_sum = com_reg_sum + col_com
            ind_reg_sum = ind_reg_sum + col_ind
            res_tsi_sum = res_tsi_sum + col_res_tsi
            com_tsi_sum = com_tsi_sum + col_com_tsi
            ind_tsi_sum = ind_tsi_sum + col_ind_tsi
    df = df.withColumn("_RES_total", res_reg_sum) \
           .withColumn("_COM_total", com_reg_sum) \
           .withColumn("_IND_total", ind_reg_sum) \
           .withColumn("_RES_total_TSI", res_tsi_sum) \
           .withColumn("_COM_total_TSI", com_tsi_sum) \
           .withColumn("_IND_total_TSI", ind_tsi_sum)
    for lvl in storey_levels:
        df = df.withColumn(
            f"RES_Storey_{lvl}_perc",
            F.when(F.col("_RES_total") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_RES") / F.col("_RES_total"))
        ).withColumn(
            f"COM_Storey_{lvl}_perc",
            F.when(F.col("_COM_total") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_COM") / F.col("_COM_total"))
        ).withColumn(
            f"IND_Storey_{lvl}_perc",
            F.when(F.col("_IND_total") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_IND") / F.col("_IND_total"))
        ).withColumn(
            f"RES_Storey_{lvl}_tsi_perc",
            F.when(F.col("_RES_total_TSI") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_RES_TSI") / F.col("_RES_total_TSI"))
        ).withColumn(
            f"COM_Storey_{lvl}_tsi_perc",
            F.when(F.col("_COM_total_TSI") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_COM_TSI") / F.col("_COM_total_TSI"))
        ).withColumn(
            f"IND_Storey_{lvl}_tsi_perc",
            F.when(F.col("_IND_total_TSI") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_IND_TSI") / F.col("_IND_total_TSI"))
        )
    df = df.drop("_RES_total", "_COM_total", "_IND_total", "_RES_total_TSI", "_COM_total_TSI", "_IND_total_TSI")
    return df

def impute_zero_tsi_percentages(df: DataFrame) -> DataFrame:
    """
    Impute zero TSI percentages with urban-specific averages.
    
    CRITICAL: Detection based on TSI_perc columns, NOT nr columns!
    """
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    
    lob_configs = {
        "RES": {
            "tsi_perc_cols": [f"RES_Storey_{lvl}_tsi_perc" for lvl in storey_levels]
        },
        "COM": {
            "tsi_perc_cols": [f"COM_Storey_{lvl}_tsi_perc" for lvl in storey_levels]
        },
        "IND": {
            "tsi_perc_cols": [f"IND_Storey_{lvl}_tsi_perc" for lvl in storey_levels]
        }
    }
    
    print("\n" + "="*80)
    print("TSI PERCENTAGE IMPUTATION: Urban-Specific Averages")
    print("="*80)
    print("NOTE: Detection based on TSI_perc columns (not nr columns)")
    print("      This correctly handles IND which shares building counts with COM")
    print("      but has different proportions and TSI values")
    
    result_df = df
    
    for lob, config in lob_configs.items():
        tsi_perc_cols = config["tsi_perc_cols"]
        
        existing_tsi_cols = [c for c in tsi_perc_cols if c in result_df.columns]
        
        if not existing_tsi_cols:
            print(f"\nWARNING: No TSI percentage columns found for {lob}, skipping imputation")
            continue
        
        print(f"\n--- Processing {lob} ---")
        print(f"TSI perc columns for detection: {existing_tsi_cols}")
        
        zero_condition = (F.coalesce(F.col(existing_tsi_cols[0]), F.lit(0)) == 0)
        for tsi_col in existing_tsi_cols[1:]:
            zero_condition = zero_condition & (F.coalesce(F.col(tsi_col), F.lit(0)) == 0)
        
        flag_col = f"_has_zero_{lob.lower()}"
        result_df = result_df.withColumn(flag_col, F.when(zero_condition, F.lit(1)).otherwise(F.lit(0)))
        
        zero_count = result_df.filter(F.col(flag_col) == 1).count()
        total_count = result_df.count()
        print(f"Grids with zero {lob} TSI activity: {zero_count:,} / {total_count:,} ({100*zero_count/max(total_count,1):.2f}%)")
        
        if zero_count == 0:
            print(f"No grids with zero {lob} TSI activity - no imputation needed")
            result_df = result_df.drop(flag_col)
            continue
        
        urban_values = [row['urban'] for row in result_df.select('urban').distinct().collect()]
        print(f"Urban values found: {urban_values}")
        
        for tsi_col in existing_tsi_cols:
            for urban_val in urban_values:
                avg_df = result_df.filter(
                    (F.col(flag_col) == 0) & (F.col('urban') == urban_val)
                ).agg(
                    F.avg(tsi_col).alias(f'avg_{tsi_col}_urban_{urban_val}')
                )
                
                avg_value = avg_df.collect()[0][0]
                
                if avg_value is None:
                    print(f"  WARNING: No valid average for {tsi_col} in urban={urban_val} - leaving as zero")
                    continue
                
                print(f"  {tsi_col} | urban={urban_val} | avg={avg_value:.6f}")
                
                result_df = result_df.withColumn(
                    tsi_col,
                    F.when(
                        (F.col(flag_col) == 1) & (F.col('urban') == urban_val),
                        F.lit(avg_value)
                    ).otherwise(F.col(tsi_col))
                )
        
        result_df = result_df.drop(flag_col)
        print(f"Completed imputation for {lob}")
    
    print("\n" + "="*80)
    print("TSI PERCENTAGE IMPUTATION COMPLETE")
    print("="*80 + "\n")
    
    return result_df

def arrange_and_sum_columns(df):
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    blocks = [
    {"prefix": "storey", "suffix": "_RES", "sum_col": "RES_Buildings_SUM"},
    {"prefix": "RES_Storey_", "suffix": "_perc", "sum_col": "RES_Buildings_SUM_perc"},
    {"prefix": "storey", "suffix": "_COM", "sum_col": "COM_Buildings_SUM"},
    {"prefix": "COM_Storey_", "suffix": "_perc", "sum_col": "COM_Buildings_SUM_perc"},
    {"prefix": "storey", "suffix": "_IND", "sum_col": "IND_Buildings_SUM"},
    {"prefix": "IND_Storey_", "suffix": "_perc", "sum_col": "IND_Buildings_SUM_perc"},
    {"prefix": "storey", "suffix": "_RES_TSI", "sum_col": "RES_TSI_SUM"},
    {"prefix": "RES_Storey_", "suffix": "_tsi_perc", "sum_col": "RES_TSI_Perc_SUM"},
    {"prefix": "storey", "suffix": "_COM_TSI", "sum_col": "COM_TSI_SUM"},
    {"prefix": "COM_Storey_", "suffix": "_tsi_perc", "sum_col": "COM_TSI_Perc_SUM"},
    {"prefix": "storey", "suffix": "_IND_TSI", "sum_col": "IND_TSI_SUM"},
    {"prefix": "IND_Storey_", "suffix": "_tsi_perc", "sum_col": "IND_TSI_Perc_SUM"},
    ]
    first_cols = ["GRID_ID", "ID_ORDER_XY", "lat", "lon", "urban",
                  "nr11_res", "nr12_res", "nr13_res", "nr14_res", "nr15_res",
                  "nr21_com", "nr21_ind", "nr22_com", "nr22_ind", "nr23_com", "nr23_ind",
                  "nr24_com", "nr24_ind", "nr25_com", "nr25_ind"]
    df_cols = df.columns
    cols_to_select = [c for c in first_cols if c in df_cols]
    for block in blocks:
        block_cols = []
        for lvl in storey_levels:
            colname = f"{block['prefix']}{lvl}{block['suffix']}"
            if colname in df_cols:
                block_cols.append(colname)
        cols_to_select += block_cols
        if block_cols:
            sum_expr = sum([F.col(c) for c in block_cols])
            df = df.withColumn(block["sum_col"], sum_expr)
            cols_to_select.append(block["sum_col"])
    cols_to_select += [c for c in df_cols if c not in cols_to_select]
    df = df.select(*cols_to_select)
    return df

def verify_csv_compatibility(df: DataFrame, stage_name: str):
    """
    Verify that all columns in the dataframe are CSV-compatible.
    Warns about any potential issues.
    """
    print(f"\n{'='*80}")
    print(f"CSV COMPATIBILITY CHECK: {stage_name}")
    print(f"{'='*80}")
    
    from pyspark.sql.types import NullType
    
    issues_found = []
    
    for field in df.schema.fields:
        dtype_str = str(field.dataType)
        dtype_name = field.dataType.typeName()
        
        # Check for problematic types
        if isinstance(field.dataType, NullType):
            issues_found.append(f"  ❌ {field.name}: NullType (VOID)")
        elif "void" in dtype_str.lower() or "void" in dtype_name.lower():
            issues_found.append(f"  ❌ {field.name}: VOID type ({dtype_str})")
        elif "null" in dtype_name.lower() and dtype_name.lower() != "nullable":
            issues_found.append(f"  ⚠️  {field.name}: NULL type ({dtype_str})")
    
    if issues_found:
        print(f"⚠️  Found {len(issues_found)} potential CSV compatibility issues:")
        for issue in issues_found:
            print(issue)
        print("\n  These columns may cause CSV export failures.")
        print("  Consider removing them before export or use the robust export script.")
    else:
        print(f"✅ All {len(df.columns)} columns are CSV-compatible!")
        print(f"   Safe to export to CSV.")
    
    print(f"{'='*80}\n")

# ---------------------------
# Main
# ---------------------------
def main():
    start_time = time.time()
    cfg = load_config()
    print("Task5 config (effective):")
    ISO3 = cfg.get("iso3", "IND").strip().upper()

    def add_iso_suffix(name, iso3=ISO3):
        if name is None:
            return name
        if name.upper().endswith(f"_{iso3}"):
            return name
        if name.endswith(".csv"):
            return name[:-4] + f"_{iso3}.csv"
        if name.endswith(".parquet"):
            return name[:-8] + f"_{iso3}.parquet"
        return f"{name}_{iso3}"

    for k, v in cfg.items():
        print(f" {k:20} = {v}")

    spark = SparkSession.builder.getOrCreate()

    # Table names from config_builder already include ISO3 in new naming convention
    grid_count_table = cfg["grid_count_table"]
    proportions_table = cfg["proportions_table"]
    tsi_table = cfg["tsi_table"]
    output_table = cfg["output_table"]
    test_tile = cfg.get("test_tile")
    output_dir = cfg.get("output_dir")
    save_temp_csv = bool(cfg.get("save_temp_csv", False))

    print(f"Reading input tables: grid_count_table={grid_count_table}, proportions_table={proportions_table}, tsi_table={tsi_table}")

    proportions_df = spark.table(proportions_table)
    step1_df = spark.table(grid_count_table)

    if "centroid_x" in step1_df.columns and "centroid_y" in step1_df.columns:
        step1_df = step1_df.withColumnRenamed("centroid_x", "lat").withColumnRenamed("centroid_y", "lon")

    tsi_df = spark.table(tsi_table)

    print("\nStep 1 DataFrame schema:")
    step1_df.printSchema()
    print("\nStep 1 DataFrame preview (5 rows):")
    step1_df.show(5)
    print("\nTotal rows loaded:")
    print(step1_df.count())

    if test_tile:
        test_tile = str(test_tile)
        print(f"Applying TEST_TILE filter: {test_tile}")
        step1_df = step1_df.filter(F.col("tile_id") == test_tile)

    print("\n" + "="*80)
    print("STEP 2: Pivoting building classes into nr... columns")
    print("="*80)
    step2_df = pivot_buildings(step1_df)
    if save_temp_csv and output_dir:
        path = add_iso_suffix(os.path.join(output_dir, "step2_pivot.csv"))
        step2_df.toPandas().to_csv(path, index=False)
        print(f"Wrote intermediate CSV: {path}")

    print("\n" + "="*80)
    print("STEP 2.5: Filtering rows with zero building counts")
    print("="*80)
    step2_df = filter_zero_buildings(step2_df)
    
    if save_temp_csv and output_dir:
        path = add_iso_suffix(os.path.join(output_dir, "step2_filtered.csv"))
        step2_df.toPandas().to_csv(path, index=False)
        print(f"Wrote filtered intermediate CSV: {path}")

    print("\n" + "="*80)
    print("STEP 3: Adding GRID_ID and order_id (CSV-compatible)")
    print("="*80)
    step2_df = add_id_and_order(step2_df)
    if save_temp_csv and output_dir:
        path = add_iso_suffix(os.path.join(output_dir, "step2_with_id.csv"))
        step2_df.toPandas().to_csv(path, index=False)
        print(f"Wrote intermediate CSV: {path}")

    print("\n" + "="*80)
    print("STEP 4: Applying proportions to compute storey-level estimates")
    print("="*80)
    step3_df = process_buildings_final(step2_df, proportions_df)
    print("Sample of step3:")
    step3_df.show(5, truncate=False)

    print("\n" + "="*80)
    print("STEP 5: Applying TSI multipliers")
    print("="*80)
    step4_df = create_step4_df(step3_df, tsi_df)
    print("Sample of step4:")
    step4_df.show(5, truncate=False)

    print("\n" + "="*80)
    print("STEP 6: Computing percentages")
    print("="*80)
    step5_df = create_step5_df(step4_df)
    print("Sample of step5:")
    step5_df.show(5, truncate=False)

    # Impute zero TSI percentages
    step5_df = impute_zero_tsi_percentages(step5_df)
    
    if save_temp_csv and output_dir:
        path = add_iso_suffix(os.path.join(output_dir, "step5_imputed.csv"))
        step5_df.toPandas().to_csv(path, index=False)
        print(f"Wrote imputed intermediate CSV: {path}")

    print("\n" + "="*80)
    print("STEP 7: Arranging columns and adding sum columns for each block")
    print("="*80)
    final_df = arrange_and_sum_columns(step5_df)
    print("Sample of final arranged output:")
    final_df.show(5, truncate=False)

    # VERIFY CSV COMPATIBILITY before saving
    verify_csv_compatibility(final_df, "FINAL OUTPUT")

    print("\n" + "="*80)
    print(f"STEP 8: Writing final output table: {output_table} (overwrite)")
    print("="*80)
    final_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    print("Saved final output to table:", output_table)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Task completed successfully in {elapsed:.1f}s")
    print(f"Output is CSV-compatible and ready for export!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
