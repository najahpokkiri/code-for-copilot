#####----------------------------------------------

## natinoal averge imputatoin

#####========================================

"""
Job 5 – Full Post-Processing Script with TSI Imputation (FINAL VERSION)
(lat/lon version: ID, GRID_ID, and output columns use `lat` and `lon`)

MODIFICATIONS:
1. Added filter to remove rows where ALL nr* columns are zero
2. Added TSI percentage imputation with urban-specific averages for categories with zero building counts
   - Uses TSI_perc columns for detection (not nr columns)
   - This correctly handles IND which doesn't have separate nr_ind columns

- Loads config, input tables, and proportions/TSI tables.
- Renames centroid_x/y to lat/lon immediately after loading input.
- Runs all main Job 5 logic: pivot, ID/order, storey-level, TSI, percentages.
- Filters out rows with all zero building counts
- **IMPUTES zero TSI percentages with urban-specific averages**
- Adds block sum columns for each output block as in diagnostics.
- Arranges columns so each block is grouped and followed by its sum.
- Saves output table.

Usage:
- Run as a Databricks notebook or Python script with Spark.
- Call with config path or set env variables as in your project.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F
# os.environ["CONFIG_PATH"] = "./config.json"
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
    
    This removes grid cells that have no building counts at all.
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
    df = step2_df
    # Create ID as lon_lat (longitude first, preserving minus signs)
    df = df.withColumn("ID", F.concat_ws("_", F.col("lon").cast("string"), F.col("lat").cast("string")))
    df = df.withColumn("GRID_ID", F.col("ID"))
    # Sort by lon ascending, then lat ascending (2-layer sort)
    window_spec = Window.orderBy(F.col("lon").asc(), F.col("lat").asc())
    df = df.withColumn("ID_ORDER_XY", F.row_number().over(window_spec))
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
        result_df = result_df.withColumn("GRID_ID", F.col("ID"))
    
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
    
    Why? IND doesn't have separate nr_ind columns - it shares building counts 
    with COM (nr21_com, nr22_com, etc.) but uses different proportions and TSI values.
    
    For each LOB (RES/COM/IND):
    1. Identify grids where ALL {LOB}_Storey_{n}_tsi_perc columns are zero
    2. Calculate urban-specific averages from grids that HAVE that LOB activity
    3. Replace zero TSI percentage values with these averages
    
    Args:
        df: DataFrame after step5 (with TSI percentages calculated)
        
    Returns:
        DataFrame with imputed TSI percentages
    """
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    
    # Define LOB configurations
    # NOTE: We use tsi_perc_cols for detection (not nr_cols)
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
        
        # Check which TSI perc columns actually exist in the dataframe
        existing_tsi_cols = [c for c in tsi_perc_cols if c in result_df.columns]
        
        if not existing_tsi_cols:
            print(f"\nWARNING: No TSI percentage columns found for {lob}, skipping imputation")
            continue
        
        print(f"\n--- Processing {lob} ---")
        print(f"TSI perc columns for detection: {existing_tsi_cols}")
        
        # Step 1: Identify rows where ALL TSI_perc columns for this LOB are zero
        # This correctly identifies grids with no activity in this LOB category
        zero_condition = (F.coalesce(F.col(existing_tsi_cols[0]), F.lit(0)) == 0)
        for tsi_col in existing_tsi_cols[1:]:
            zero_condition = zero_condition & (F.coalesce(F.col(tsi_col), F.lit(0)) == 0)
        
        # Add flag column to identify zero rows
        flag_col = f"_has_zero_{lob.lower()}"
        result_df = result_df.withColumn(flag_col, F.when(zero_condition, F.lit(1)).otherwise(F.lit(0)))
        
        # Count how many rows have zero for this LOB
        zero_count = result_df.filter(F.col(flag_col) == 1).count()
        total_count = result_df.count()
        print(f"Grids with zero {lob} TSI activity: {zero_count:,} / {total_count:,} ({100*zero_count/max(total_count,1):.2f}%)")
        
        if zero_count == 0:
            print(f"No grids with zero {lob} TSI activity - no imputation needed")
            result_df = result_df.drop(flag_col)
            continue
        
        # Step 2: Calculate urban-specific averages for each TSI percentage column
        # Only from grids that HAVE this LOB (flag = 0)
        
        # Get distinct urban values
        urban_values = [row['urban'] for row in result_df.select('urban').distinct().collect()]
        print(f"Urban values found: {urban_values}")
        
        for tsi_col in existing_tsi_cols:
            # For each urban type, calculate average from non-zero grids
            for urban_val in urban_values:
                # Calculate average from grids that have this LOB and match urban type
                avg_df = result_df.filter(
                    (F.col(flag_col) == 0) & (F.col('urban') == urban_val)
                ).agg(
                    F.avg(tsi_col).alias(f'avg_{tsi_col}_urban_{urban_val}')
                )
                
                avg_value = avg_df.collect()[0][0]
                
                if avg_value is None:
                    print(f"  WARNING: No valid average for {tsi_col} in urban={urban_val} - leaving as zero")
                    print(f"           This means NO grids with urban={urban_val} have {lob} TSI activity!")
                    continue
                
                print(f"  {tsi_col} | urban={urban_val} | avg={avg_value:.6f}")
                
                # Replace zero values with average for matching urban type
                result_df = result_df.withColumn(
                    tsi_col,
                    F.when(
                        (F.col(flag_col) == 1) & (F.col('urban') == urban_val),
                        F.lit(avg_value)
                    ).otherwise(F.col(tsi_col))
                )
        
        # Drop the flag column
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

    grid_count_table = add_iso_suffix(cfg["grid_count_table"])
    proportions_table = add_iso_suffix(cfg["proportions_table"])
    tsi_table = add_iso_suffix(cfg["tsi_table"])
    output_table = add_iso_suffix(cfg["output_table"])
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
    print("STEP 3: Adding GRID_ID and order_id")
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

    # ====================================================================
    # STEP 6.5: Impute zero TSI percentages with urban-specific averages
    # ====================================================================
    step5_df = impute_zero_tsi_percentages(step5_df)
    
    if save_temp_csv and output_dir:
        path = add_iso_suffix(os.path.join(output_dir, "step5_imputed.csv"))
        step5_df.toPandas().to_csv(path, index=False)
        print(f"Wrote imputed intermediate CSV: {path}")
    # ====================================================================

    print("\n" + "="*80)
    print("STEP 7: Arranging columns and adding sum columns for each block")
    print("="*80)
    final_df = arrange_and_sum_columns(step5_df)
    print("Sample of final arranged output:")
    final_df.show(5, truncate=False)

    print("\n" + "="*80)
    print(f"STEP 8: Writing final output table: {output_table} (overwrite)")
    print("="*80)
    final_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    print("Saved final output to table:", output_table)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Task completed successfully in {elapsed:.1f}s")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
