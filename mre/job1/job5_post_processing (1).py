"""
Job 5 — Full Post-Processing Script with Column Arrangement and Sums
(lat/lon version: ID, GRID_ID, and output columns use `lat` and `lon`)

- Loads config, input tables, and proportions/TSI tables.
- Renames centroid_x/y to lat/lon immediately after loading input.
- Runs all main Job 5 logic: pivot, ID/order, storey-level, TSI, percentages.
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
from typing import Dict, Any
from pyspark.sql import SparkSession, DataFrame, Window
import pyspark.sql.functions as F

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

# os.environ["CONFIG_PATH"] = "./config.json"
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
    res_df = df.withColumn("LOB",
                F.when(F.col("built").between(11,15), F.lit("RES"))
                 .when(F.col("built").between(21,25), F.lit("COM")))
    ind_dup = df.filter(F.col("built").between(21,25)).withColumn("LOB", F.lit("IND"))
    unioned = res_df.unionByName(ind_dup, allowMissingColumns=True)
    with_built_lob = unioned.withColumn(
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

# def add_id_and_order(step2_df: DataFrame) -> DataFrame:
#     df = step2_df
#     # Use lat/lon for ID construction
#     # df = df.withColumn("ID", F.concat_ws("", F.col("lat").cast("lon"), F.col("lon").cast("lon")))
#     df = df.withColumn("ID", F.concat_ws("", F.col("lat").cast("double"), F.col("lon").cast("double")))
#     df = df.withColumn("GRID_ID", F.col("ID"))
#     window_spec = Window.orderBy(F.col("ID").cast("long"))
#     df = df.withColumn("order_id", F.row_number().over(window_spec))
#     return df
def add_id_and_order(step2_df: DataFrame) -> DataFrame:
    df = step2_df

    # Multiply by 1e6 (or more for more precision), cast to int, then concat as strings
    lat_int = (F.col("lat") * 1e6).cast("long")
    lon_int = (F.col("lon") * 1e6).cast("long")
    df = df.withColumn("ID", F.concat_ws("_", lat_int, lon_int))
    df = df.withColumn("GRID_ID", F.col("ID"))
    window_spec = Window.orderBy(F.col("ID"))
    df = df.withColumn("order_id", F.row_number().over(window_spec))
    return df
    
def process_buildings_final(step2_df: DataFrame, props_df: DataFrame) -> DataFrame:
    result_df = step2_df
    storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
    nr_cols = [c for c in result_df.columns if c.startswith("nr")]
    for type_name, building_range, suffix in [
        ("RES", range(11, 16), "res"),
        ("COM", range(21, 26), "com"),
        ("IND", range(21, 26), "ind")
    ]:
        for building in building_range:
            props = props_df.filter(F.col("storey") == building).select("urban", *[F.col(f"`{s}`").alias(f"prop_{s}") for s in storey_levels])
            result_df = result_df.join(props, on="urban", how="left")
            for storey in storey_levels:
                nr_col = f"nr{building}_{suffix}"
                prop_col = f"prop_{storey}"
                out_col = f"storey{storey}_{suffix}_{building}"
                if nr_col not in result_df.columns:
                    result_df = result_df.withColumn(nr_col, F.lit(0))
                result_df = result_df.withColumn(out_col, (F.coalesce(F.col(nr_col), F.lit(0)) * F.coalesce(F.col(prop_col), F.lit(0.0))))
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
    select_cols = ["GRID_ID","order_id","lat", "lon", "urban"] + nr_cols + final_cols
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
    first_cols = ["GRID_ID", "order_id", "lat", "lon", "urban",
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
        """
        Appends _ISO3 to a table or file name if not already present.
        Handles .csv/.parquet endings for files.
        """
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
    # grid_count_table = cfg["grid_count_table"]
    # proportions_table = cfg["proportions_table"]
    # tsi_table = cfg["tsi_table"]

    grid_count_table = add_iso_suffix(cfg["grid_count_table"])
    proportions_table = add_iso_suffix(cfg["proportions_table"])
    tsi_table = add_iso_suffix(cfg["tsi_table"])
    # output_table = cfg["output_table"]
    output_table = add_iso_suffix(cfg["output_table"])
    test_tile = cfg.get("test_tile")
    output_dir = cfg.get("output_dir")
    save_temp_csv = bool(cfg.get("save_temp_csv", False))

    print(f"Reading input tables: grid_count_table={grid_count_table}, proportions_table={proportions_table}, tsi_table={tsi_table}")

    proportions_df = spark.table(proportions_table)
    step1_df = spark.table(grid_count_table)

    # --- Rename centroid_x, centroid_y to lat, lon immediately after loading ---
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

    print("Pivoting building classes into nr... columns")
    step2_df = pivot_buildings(step1_df)
    if save_temp_csv and output_dir:
        # path = os.path.join(output_dir, "step2_pivot.csv")
        path = add_iso_suffix(os.path.join(output_dir, "step2_pivot.csv"))
        step2_df.toPandas().to_csv(path, index=False)
        print(f"Wrote intermediate CSV: {path}")

    print("Adding GRID_ID and order_id")
    step2_df = add_id_and_order(step2_df)
    if save_temp_csv and output_dir:
        # path = os.path.join(output_dir, "step2_with_id.csv")
        path = add_iso_suffix(os.path.join(output_dir, "step2_with_id.csv"))
        step2_df.toPandas().to_csv(path, index=False)
        print(f"Wrote intermediate CSV: {path}")

    print("Applying proportions to compute storey-level estimates")
    step3_df = process_buildings_final(step2_df, proportions_df)
    print("Sample of step3:")
    step3_df.show(5, truncate=False)

    print("Applying TSI multipliers")
    step4_df = create_step4_df(step3_df, tsi_df)
    print("Sample of step4:")
    step4_df.show(5, truncate=False)

    print("Computing percentages")
    step5_df = create_step5_df(step4_df)
    print("Sample of step5:")
    step5_df.show(5, truncate=False)

    # ---- Column arrangement and sum blocks ----
    print("Arranging columns and adding sum columns for each block")
    final_df = arrange_and_sum_columns(step5_df)
    print("Sample of final arranged output:")
    final_df.show(5, truncate=False)

    print(f"Writing final output table: {output_table} (overwrite)")
    final_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
    print("Saved final output to table:", output_table)

    elapsed = time.time() - start_time
    print(f"Task completed in {elapsed:.1f}s")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Task 5 — Post processing (runnable script)

# Converts the notebook steps into a runnable script that:
# - loads config (JSON) from CONFIG_PATH (dbfs:/ or local) or uses env overrides,
# - reads required Spark tables,
# - performs transformations tested in notebook:
#   * pivot built class counts by LOB (RES/COM/IND)
#   * create GRID_ID and order_id
#   * apply proportions to produce storey-level estimates
#   * apply TSI multipliers
#   * compute percentages and reorder columns
# - writes final table to Hive/Delta table (output_table)

# Usage (Databricks Job):
#  - Upload a config JSON to DBFS (e.g. dbfs:/configs/task5_config.json)
#  - Pass single job parameter to the task:
#      ["--config_path","dbfs:/configs/task5_config.json"]
#  - Optional overrides can be passed as additional parameters (they map to env vars).

# Run interactively in a notebook:
#  - Set os.environ["CONFIG_PATH"] = "dbfs:/tmp/task5_config.json"
#  - import and call main() or run the script.

# Note: This script assumes SparkSession is available and the input tables referenced in config exist.
# """

# import os
# import sys
# import json
# import time
# import traceback
# from typing import Dict, Any, List

# from pyspark.sql import SparkSession, DataFrame, Window
# import pyspark.sql.functions as F

# # ---------------------------
# # CLI -> env wrapper
# # ---------------------------
# if len(sys.argv) > 1:
#     args = sys.argv[1:]
#     i = 0
#     while i < len(args):
#         key = args[i]
#         if key.startswith("--"):
#             env_key = key.lstrip("-").upper()
#             value = ""
#             if (i + 1) < len(args) and not args[i + 1].startswith("--"):
#                 value = args[i + 1]
#                 i += 2
#             else:
#                 i += 1
#             if value != "":
#                 os.environ[env_key] = value
#         else:
#             i += 1
# # os.environ["CONFIG_PATH"] = "./config.json"
# # ---------------------------
# # Config loader & defaults
# # ---------------------------
# DEFAULT_CONFIG: Dict[str, Any] = {
#     "grid_count_table": None,          # e.g. "prp_mr_bdap_projects.geospatialsolutions.grid_counts_ind"
#     "proportions_table": None,         # e.g. "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_proportions_input"
#     "tsi_table": None,                 # e.g. "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_tsi_input"
#     "output_table": None,              # e.g. "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_output"
#     "test_tile": None,                 # optional single tile id to run on
#     "save_temp_csv": False,            # optional: write intermediate CSVs to OUTPUT_DIR if provided
#     "output_dir": None                 # optional: path to save CSVs for inspection (e.g. /dbfs/tmp/)
# }

# def _read_json_path(path: str) -> Dict[str, Any]:
#     if path.startswith("dbfs:"):
#         local_path = path.replace("dbfs:", "/dbfs", 1)
#     else:
#         local_path = path
#     with open(local_path, "r") as f:
#         return json.load(f)

# def load_config() -> Dict[str, Any]:
#     cfg = dict(DEFAULT_CONFIG)
#     cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG", "") or os.environ.get("CONFIG_JSON", "")
#     if cfg_path:
#         try:
#             loaded = _read_json_path(cfg_path)
#             if not isinstance(loaded, dict):
#                 raise ValueError("config file must be a JSON object")
#             cfg.update(loaded)
#         except Exception as e:
#             raise RuntimeError(f"Failed to load config file '{cfg_path}': {e}")

#     # allow simple overrides via environment variables
#     for key in list(cfg.keys()):
#         env_key = key.upper()
#         if env_key in os.environ and os.environ[env_key] != "":
#             val = os.environ[env_key]
#             # boolean parse
#             if isinstance(cfg[key], bool):
#                 cfg[key] = str(val).lower() in ("true","1","t","yes")
#             else:
#                 cfg[key] = val
#     # minimal validation
#     required = ["grid_count_table","proportions_table","tsi_table","output_table"]
#     missing = [k for k in required if not cfg.get(k)]
#     if missing:
#         raise RuntimeError(f"Missing required config keys: {missing}. Please add them to config or pass overrides.")
#     return cfg

# # ---------------------------
# # Processing functions
# # ---------------------------
# def pivot_buildings(step1_df: DataFrame) -> DataFrame:
#     """
#     From the raw lon built rows, produce a grid-level pivot table with nr{class}_{lob} columns.
#     This follows the notebook logic:
#       - mark LOB as RES for built 11-15, COM for 21-25
#       - duplicate 21-25 rows but mark as IND as well
#       - create built_lob key and pivot on it taking first(count) and filling null with 0
#     """
#     # create LOB column and union to add IND duplicates for 21-25
#     # note: step1_df expected columns: grid_id, built, count, urban, centroid_x, centroid_y, ...
#     df = step1_df

#     res_df = df.withColumn("LOB",
#                 F.when(F.col("built").between(11,15), F.lit("RES"))
#                  .when(F.col("built").between(21,25), F.lit("COM"))
#             )

#     ind_dup = df.filter(F.col("built").between(21,25)).withColumn("LOB", F.lit("IND"))

#     unioned = res_df.unionByName(ind_dup, allowMissingColumns=True)

#     with_built_lob = unioned.withColumn(
#         "built_lob",
#         F.concat(F.lit("nr"), F.col("built").cast("string"), F.lit("_"), F.lower(F.col("LOB")))
#     )

#     # group and pivot
#     pivoted = (with_built_lob
#                .groupBy("grid_id", "urban", "centroid_x", "centroid_y")
#                .pivot("built_lob")
#                .agg(F.first("count"))
#                .na.fill(0)
#                )
#     return pivoted

# def add_id_and_order(step2_df: DataFrame) -> DataFrame:
#     """
#     Create ID by concatenating centroid_x and centroid_y cast to lon,
#     then create GRID_ID and order_id (row_number by ID).
#     """
#     df = step2_df
#     df = df.withColumn("ID", F.concat_ws("", F.col("centroid_x").cast("lon"), F.col("centroid_y").cast("lon")))
#     # create GRID_ID as string (same as ID) and order_id as row_number
#     df = df.withColumn("GRID_ID", F.col("ID"))
#     window_spec = Window.orderBy(F.col("ID").cast("lon"))
#     df = df.withColumn("order_id", F.row_number().over(window_spec))
#     return df

# def process_buildings_final(step2_df: DataFrame, props_df: DataFrame) -> DataFrame:
#     """
#     Apply the proportions to produce per-storey type estimates as in the notebook.
#     The notebook's approach was to loop over building classes and join the proportions by urban+storey.
#     This implementation follows that approach but uses small joins (proportions table expected to be small).
#     """
#     result_df = step2_df
#     storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]

#     # gather nr columns (those created by pivot)
#     nr_cols = [c for c in result_df.columns if c.startswith("nr")]

#     # We'll loop the same way as notebook: for each building range and suffix, join the relevant props row
#     for type_name, building_range, suffix in [
#         ("RES", range(11, 16), "res"),
#         ("COM", range(21, 26), "com"),
#         ("IND", range(21, 26), "ind")
#     ]:
#         for building in building_range:
#             # get props for this building (small table row)
#             props = props_df.filter(F.col("storey") == building).select("urban", *[F.col(f"`{s}`").alias(f"prop_{s}") for s in storey_levels])
#             # left join by urban; props is expected to be small
#             result_df = result_df.join(props, on="urban", how="left")
#             # multiply nr columns by proportions to create intermediate storey{lvl}_{suffix}_{building}
#             for storey in storey_levels:
#                 nr_col = f"nr{building}_{suffix}"
#                 prop_col = f"prop_{storey}"
#                 out_col = f"storey{storey}_{suffix}_{building}"
#                 # if nr_col missing (no such building in pivot), treat as 0
#                 if nr_col not in result_df.columns:
#                     result_df = result_df.withColumn(nr_col, F.lit(0))
#                 result_df = result_df.withColumn(out_col, (F.coalesce(F.col(nr_col), F.lit(0)) * F.coalesce(F.col(prop_col), F.lit(0.0))))
#             # drop prop_* columns introduced by this join
#             drop_props = [c for c in result_df.columns if c.startswith("prop_")]
#             if drop_props:
#                 result_df = result_df.drop(*drop_props)

#         # After processing this suffix, sum across buildings to get storey{lvl}_{type_name}
#         for storey in storey_levels:
#             summands = [f"coalesce(storey{storey}_{suffix}_{i}, 0)" for i in building_range]
#             sum_expr = " + ".join(summands)
#             result_df = result_df.withColumn(f"storey{storey}_{type_name}", F.expr(sum_expr))
#         # drop intermediate columns matching pattern _{suffix}_ (cleanup)
#         drop_intermediates = [c for c in result_df.columns if f"_{suffix}_" in c]
#         if drop_intermediates:
#             result_df = result_df.drop(*drop_intermediates)

#     # Build final column list: GRID_ID, urban, nr_cols, and storey totals
#     final_cols = []
#     for storey in storey_levels:
#         final_cols.extend([f"storey{storey}_RES", f"storey{storey}_COM", f"storey{storey}_IND"])

#     # ensure GRID_ID exists (should be created earlier)
#     if "GRID_ID" not in result_df.columns and "ID" in result_df.columns:
#         result_df = result_df.withColumn("GRID_ID", F.col("ID"))

#     select_cols = ["GRID_ID","order_id","centroid_x", "centroid_y", "urban"] + nr_cols + final_cols
#     # guard for missing columns
#     select_cols = [c for c in select_cols if c in result_df.columns]
#     return result_df.select(*select_cols)

# def create_step4_df(step3_df: DataFrame, tsi_df: DataFrame) -> DataFrame:
#     """
#     Multiply storey-level counts by TSI multipliers from tsi_df.
#     The tsi_df expected structure: rows for LOB values with columns matching storey_levels.
#     We'll collect tsi_df to the driver (it should be small).
#     """
#     df = step3_df
#     # collect tsi table into a dict: {LOB: row_as_dict}
#     tsi_rows = tsi_df.collect()
#     tsi_map = {}
#     for r in tsi_rows:
#         tsi_map[r["LOB"]] = r.asDict()

#     storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]
#     for storey in storey_levels:
#         # RES
#         res_val = tsi_map.get("RES", {}).get(storey, 0)
#         df = df.withColumn(f"storey{storey}_RES_TSI", F.col(f"storey{storey}_RES") * F.lit(res_val))
#         # COM
#         com_val = tsi_map.get("COM", {}).get(storey, 0)
#         df = df.withColumn(f"storey{storey}_COM_TSI", F.col(f"storey{storey}_COM") * F.lit(com_val))
#         # IND
#         ind_val = tsi_map.get("IND", {}).get(storey, 0)
#         df = df.withColumn(f"storey{storey}_IND_TSI", F.col(f"storey{storey}_IND") * F.lit(ind_val))
#     return df

# def create_step5_df(step4_df: DataFrame) -> DataFrame:
#     """
#     Computes denominators and percentages, reorders columns as in notebook.
#     """
#     df = step4_df
#     storey_levels = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]

#     # compute totals
#     res_reg_sum = None
#     com_reg_sum = None
#     ind_reg_sum = None
#     res_tsi_sum = None
#     com_tsi_sum = None
#     ind_tsi_sum = None

#     for i, lvl in enumerate(storey_levels):
#         col_res = F.col(f"storey{lvl}_RES")
#         col_com = F.col(f"storey{lvl}_COM")
#         col_ind = F.col(f"storey{lvl}_IND")

#         col_res_tsi = F.col(f"storey{lvl}_RES_TSI")
#         col_com_tsi = F.col(f"storey{lvl}_COM_TSI")
#         col_ind_tsi = F.col(f"storey{lvl}_IND_TSI")

#         if i == 0:
#             res_reg_sum = col_res
#             com_reg_sum = col_com
#             ind_reg_sum = col_ind
#             res_tsi_sum = col_res_tsi
#             com_tsi_sum = col_com_tsi
#             ind_tsi_sum = col_ind_tsi
#         else:
#             res_reg_sum = res_reg_sum + col_res
#             com_reg_sum = com_reg_sum + col_com
#             ind_reg_sum = ind_reg_sum + col_ind
#             res_tsi_sum = res_tsi_sum + col_res_tsi
#             com_tsi_sum = com_tsi_sum + col_com_tsi
#             ind_tsi_sum = ind_tsi_sum + col_ind_tsi

#     df = df.withColumn("_RES_total", res_reg_sum) \
#            .withColumn("_COM_total", com_reg_sum) \
#            .withColumn("_IND_total", ind_reg_sum) \
#            .withColumn("_RES_total_TSI", res_tsi_sum) \
#            .withColumn("_COM_total_TSI", com_tsi_sum) \
#            .withColumn("_IND_total_TSI", ind_tsi_sum)

#     # create percentage columns
#     for lvl in storey_levels:
#         df = df.withColumn(
#             f"RES_Storey_{lvl}_perc",
#             F.when(F.col("_RES_total") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_RES") / F.col("_RES_total"))
#         ).withColumn(
#             f"COM_Storey_{lvl}_perc",
#             F.when(F.col("_COM_total") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_COM") / F.col("_COM_total"))
#         ).withColumn(
#             f"IND_Storey_{lvl}_perc",
#             F.when(F.col("_IND_total") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_IND") / F.col("_IND_total"))
#         ).withColumn(
#             f"RES_Storey_{lvl}_tsi_perc",
#             F.when(F.col("_RES_total_TSI") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_RES_TSI") / F.col("_RES_total_TSI"))
#         ).withColumn(
#             f"COM_Storey_{lvl}_tsi_perc",
#             F.when(F.col("_COM_total_TSI") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_COM_TSI") / F.col("_COM_total_TSI"))
#         ).withColumn(
#             f"IND_Storey_{lvl}_tsi_perc",
#             F.when(F.col("_IND_total_TSI") == 0, F.lit(0.0)).otherwise(F.col(f"storey{lvl}_IND_TSI") / F.col("_IND_total_TSI"))
#         )

#     df = df.drop("_RES_total", "_COM_total", "_IND_total", "_RES_total_TSI", "_COM_total_TSI", "_IND_total_TSI")

#     # Reorder columns similar to notebook: after each storey{lvl}_RES insert RES/COM/IND perc, then append TSI perc block
#     original_cols = step4_df.columns
#     tsi_columns = [c for c in original_cols if c.endswith("_TSI")]
#     reg_perc_cols = []
#     tsi_perc_cols = []
#     for lvl in storey_levels:
#         reg_perc_cols.extend([f"RES_Storey_{lvl}_perc", f"COM_Storey_{lvl}_perc", f"IND_Storey_{lvl}_perc"])
#         tsi_perc_cols.extend([f"RES_Storey_{lvl}_tsi_perc", f"COM_Storey_{lvl}_tsi_perc", f"IND_Storey_{lvl}_tsi_perc"])

#     new_col_order = []
#     for c in original_cols:
#         new_col_order.append(c)
#         if c.startswith("storey") and c.endswith("_RES"):
#             lvl = c[len("storey") : -len("_RES")]
#             # append perc cols
#             new_col_order.append(f"RES_Storey_{lvl}_perc")
#             new_col_order.append(f"COM_Storey_{lvl}_perc")
#             new_col_order.append(f"IND_Storey_{lvl}_perc")
#         if tsi_columns and c == tsi_columns[-1]:
#             new_col_order.extend(tsi_perc_cols)

#     # ensure uniqueness and presence
#     seen = set()
#     final_cols = []
#     for c in new_col_order:
#         if c not in seen and c in df.columns:
#             seen.add(c)
#             final_cols.append(c)
#     for c in df.columns:
#         if c not in seen:
#             final_cols.append(c)
#     df = df.select(*final_cols)
#     return df

# # ---------------------------
# # Main
# # ---------------------------
# def main():
#     start_time = time.time()
#     cfg = load_config()
#     print("Task5 config (effective):")
#     for k, v in cfg.items():
#         # hide big objects if any
#         print(f" {k:20} = {v}")

#     spark = SparkSession.builder.getOrCreate()

#     grid_count_table = cfg["grid_count_table"]
#     proportions_table = cfg["proportions_table"]
#     tsi_table = cfg["tsi_table"]
#     output_table = cfg["output_table"]
#     test_tile = cfg.get("test_tile")
#     output_dir = cfg.get("output_dir")
#     save_temp_csv = bool(cfg.get("save_temp_csv", False))

#     print(f"Reading input tables: grid_count_table={grid_count_table}, proportions_table={proportions_table}, tsi_table={tsi_table}")

#     # Read inputs
#     proportions_df = spark.table(proportions_table)
#     step1_df = spark.table(grid_count_table)
#     tsi_df = spark.table(tsi_table)

#     print("\nStep 1 DataFrame schema:")
#     step1_df.printSchema()
#     print("\nStep 1 DataFrame preview (5 rows):")
#     step1_df.show(5)
#     print("\nTotal rows loaded:")
#     print(step1_df.count())

#     # Optionally restrict to one tile for quick tests
#     if test_tile:
#         test_tile = str(test_tile)
#         print(f"Applying TEST_TILE filter: {test_tile}")
#         step1_df = step1_df.filter(F.col("tile_id") == test_tile)

#     # Step 1 -> Step 2: pivot to nr columns
#     print("Pivoting building classes into nr... columns")
#     step2_df = pivot_buildings(step1_df)
#     if save_temp_csv and output_dir:
#         path = os.path.join(output_dir, "step2_pivot.csv")
#         step2_df.toPandas().to_csv(path, index=False)
#         print(f"Wrote intermediate CSV: {path}")

#     # Step 2: create ID and order_id (GRID_ID)
#     print("Adding GRID_ID and order_id")
#     step2_df = add_id_and_order(step2_df)
#     if save_temp_csv and output_dir:
#         path = os.path.join(output_dir, "step2_with_id.csv")
#         step2_df.toPandas().to_csv(path, index=False)
#         print(f"Wrote intermediate CSV: {path}")

#     # Step 3: apply proportions
#     print("Applying proportions to compute storey-level estimates")
#     step3_df = process_buildings_final(step2_df, proportions_df)
#     print("Sample of step3:")
#     step3_df.show(5, truncate=False)

#     # Step 4: apply TSI multipliers
#     print("Applying TSI multipliers")
#     step4_df = create_step4_df(step3_df, tsi_df)
#     print("Sample of step4:")
#     step4_df.show(5, truncate=False)

#     # Step 5: compute percentages and reorder columns
#     print("Computing percentages and reordering columns")
#     step5_df = create_step5_df(step4_df)
#     print("Sample of final (step5):")
#     step5_df.show(5, truncate=False)

#     # Write final table
#     print(f"Writing final output table: {output_table} (overwrite)")
#     try:
#         step5_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(output_table)
#         print("Saved final output to table:", output_table)
#     except Exception as e:
#         tb = traceback.format_exc()
#         print("ERROR writing final output:", e, tb)
#         raise

#     elapsed = time.time() - start_time
#     print(f"Task completed in {elapsed:.1f}s")

# if __name__ == "__main__":
#     main()
