#!/usr/bin/env python3
"""
Task 5b — Create per-LOB TSI proportion views using *_tsi_perc columns and level_mapping bin renaming.

- ISO3 is the only dynamic value, read from config or env (default "IND").
- All table/view names are auto-computed with the ISO3 suffix; nothing else is needed from config.
"""

import os
import sys
import json
import traceback
import re
from typing import Dict, List, Optional

from pyspark.sql import SparkSession
# os.environ["CONFIG_PATH"] = "./config.json"
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
ORDER_COL_CANDIDATES = ["order_id","ORDER_ID_XY","orderid"]
DROP_TABLE_IF_EXISTS = False
FALLBACK_VIEW_SUFFIX = "_view"

def get_iso3():
    # Try config file, then env, then default
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
    select_parts.append(f"`{actual_id}` AS ID" if actual_id in cols else "NULL AS ID")
    select_parts.append(f"`{actual_x}` AS POINT_X" if actual_x else "NULL AS POINT_X")
    select_parts.append(f"`{actual_y}` AS POINT_Y" if actual_y else "NULL AS POINT_Y")
    select_parts.append(f"`{actual_order}` AS ORDER_ID_XY" if actual_order else "NULL AS ORDER_ID_XY")

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
    print(f"\nCreating view: {fq_view}\n{create_sql}\n")
    spark.sql(create_sql)
    print(f"Created view: {fq_view}")

def main():
    try:
        iso3 = get_iso3()
        input_table = add_iso_suffix(BASE_INPUT_TABLE, iso3)
        output_view_base = add_iso_suffix(BASE_OUTPUT_VIEW, iso3)
        print(f"ISO3: {iso3}")
        print(f"Input table: {input_table}")
        print(f"Output view base: {output_view_base}")
        spark = SparkSession.builder.getOrCreate()
        cols = spark.table(input_table).columns
        for lob in LOBS:
            create_view_for_lob(spark, input_table, output_view_base, cols, lob, iso3)
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
# #
# #!/usr/bin/env python3
# """
# Task 5b — Create per-LOB TSI proportion views using *_tsi_perc columns and level_mapping bin renaming.

# For each LOB (RES/COM/IND):
#  - Select all columns starting with LOB_ and ending with _tsi_perc.
#  - Extract the storey level (e.g., '4_5') from the column.
#  - Rename each column to its mapped bin (from config.level_mapping), e.g., '4_5' → '5', etc.
#  - Include ID, POINT_X, POINT_Y, ORDER_ID_XY (from candidates or NULL if not found).
#  - SUM column is the sum of only the mapped bins present in the view.

# If the desired view name exists as a table, append fallback_view_suffix (default "_view").

# Config example:
# {
#   "input_table": "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_output",
#   "output_view_base": "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_tsi_proportions",
#   "lobs": ["RES","COM","IND"],
#   "level_mapping": {
#     "1": "1",
#     "2": "2",
#     "3": "3",
#     "4_5": "5",
#     "6_8": "7",
#     "9_20": "10",
#     "20": "40"
#   },
#   "x_col_candidates": ["centroid_x","lon","POINT_X","longitude"],
#   "y_col_candidates": ["centroid_y","lat","POINT_Y","latitude"],
#   "order_col_candidates": ["order_id","ORDER_ID_XY","orderid"],
#   "drop_table_if_exists": false,
#   "fallback_view_suffix": "_view"
# }
# """

# import os
# import sys
# import json
# import traceback
# import re
# from typing import Dict, List, Optional

# from pyspark.sql import SparkSession
# os.environ["CONFIG_PATH"] = "./views_config.json"
# DEFAULT_CFG = {
#     "input_table": None,
#     "output_view_base": None,
#     "lobs": ["RES", "COM", "IND"],
#     "level_mapping": {
#         "1": "1",
#         "2": "2",
#         "3": "3",
#         "4_5": "5",
#         "6_8": "7",
#         "9_20": "10",
#         "20": "40"
#     },
#     "x_col_candidates": ["centroid_x","lon","POINT_X","longitude"],
#     "y_col_candidates": ["centroid_y","lat","POINT_Y","latitude"],
#     "order_col_candidates": ["order_id","ORDER_ID_XY","orderid"],
#     "drop_table_if_exists": False,
#     "fallback_view_suffix": "_view"
# }

# def read_config() -> Dict:
#     cfg = dict(DEFAULT_CFG)
#     p = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG")
#     if p:
#         if p.startswith("dbfs:"):
#             p = p.replace("dbfs:", "/dbfs", 1)
#         with open(p, "r", encoding="utf8") as fh:
#             j = json.load(fh)
#         cfg.update(j)
#     for k in list(cfg.keys()):
#         ek = k.upper()
#         if ek in os.environ and os.environ[ek] != "":
#             val = os.environ[ek]
#             try:
#                 cfg[k] = json.loads(val)
#             except Exception:
#                 if isinstance(cfg[k], bool):
#                     cfg[k] = str(val).lower() in ("true","1","t","yes")
#                 elif isinstance(cfg[k], list):
#                     cfg[k] = [s.strip() for s in str(val).split(",") if s.strip()]
#                 else:
#                     cfg[k] = val
#     if not cfg.get("input_table") or not cfg.get("output_view_base"):
#         raise RuntimeError("input_table and output_view_base required in config")
#     return cfg

# def find_first(cols: List[str], candidates: List[str]) -> Optional[str]:
#     upcols = [c.upper() for c in cols]
#     for cand in candidates:
#         for idx, uc in enumerate(upcols):
#             if uc == cand.upper():
#                 return cols[idx]
#     for cand in candidates:
#         low = cand.lower()
#         for idx, c in enumerate(cols):
#             if low in c.lower():
#                 return cols[idx]
#     return None

# def create_view_for_lob(spark, cfg, cols, lob):
#     input_table = cfg["input_table"]
#     out_base = cfg["output_view_base"]
#     x_cands = cfg["x_col_candidates"]
#     y_cands = cfg["y_col_candidates"]
#     order_cands = cfg["order_col_candidates"]
#     drop_if_table = bool(cfg.get("drop_table_if_exists", False))
#     fallback_suffix = cfg.get("fallback_view_suffix", "_view")
#     level_map = cfg.get("level_mapping", {})

#     # Collect coordinate/order columns
#     actual_x = find_first(cols, x_cands)
#     actual_y = find_first(cols, y_cands)
#     actual_order = find_first(cols, order_cands)
#     actual_id = find_first(cols, ["GRID_ID","ID","grid_id","id"]) or "GRID_ID"

#     # Find and map bin columns
#     lob_cols = [c for c in cols if c.upper().startswith(lob.upper()+"_") and c.lower().endswith("_tsi_perc")]
#     bin_map = {}
#     for c in lob_cols:
#         m = re.match(rf"{lob}_(Storey_)?(.+)_tsi_perc", c, re.IGNORECASE)
#         if m:
#             orig_level = m.group(2)
#             mapped_bin = level_map.get(orig_level, orig_level)
#             bin_map[orig_level] = (mapped_bin, c)
#     # Only one column per mapped_bin (first occurrence)
#     mapped_cols = {}
#     for orig, (mapped_bin, c) in bin_map.items():
#         if mapped_bin not in mapped_cols:
#             mapped_cols[mapped_bin] = c

#     # Sorted order as per mapping values
#     bin_order = [v for k,v in level_map.items()]
#     select_parts = []
#     select_parts.append(f"`{actual_id}` AS ID" if actual_id in cols else "NULL AS ID")
#     select_parts.append(f"`{actual_x}` AS POINT_X" if actual_x else "NULL AS POINT_X")
#     select_parts.append(f"`{actual_y}` AS POINT_Y" if actual_y else "NULL AS POINT_Y")
#     select_parts.append(f"`{actual_order}` AS ORDER_ID_XY" if actual_order else "NULL AS ORDER_ID_XY")

#     for b in bin_order:
#         c = mapped_cols.get(b)
#         if c:
#             select_parts.append(f"COALESCE(`{c}`, 0.0) AS `{b}`")
#         else:
#             select_parts.append(f"0.0 AS `{b}`")

#     # SUM = sum of mapped bin columns in order
#     sum_expr = " + ".join([f"`{b}`" for b in bin_order])
#     select_parts.append(f"({sum_expr}) AS SUM")

#     select_sql = ",\n  ".join(select_parts)
#     short_view = f"{out_base}_{lob.lower()}"
#     fq_view = short_view

#     # Check for table collision (fallback if needed)
#     try:
#         schemaPrefix = ".".join(fq_view.split(".")[:-1])
#         short_name = fq_view.split(".")[-1].lower()
#         existing_tables = [r["tableName"].lower() for r in spark.sql(f"SHOW TABLES IN {schemaPrefix}").collect()] if "." in schemaPrefix else []
#     except Exception:
#         existing_tables = []
#     if short_name in existing_tables and not drop_if_table:
#         fq_view = fq_view + fallback_suffix

#     create_sql = f"CREATE OR REPLACE VIEW {fq_view} AS\nSELECT\n  {select_sql}\nFROM {input_table}"
#     print(f"\nCreating view: {fq_view}\n{create_sql}\n")
#     spark.sql(create_sql)
#     print(f"Created view: {fq_view}")

# def main():
#     try:
#         cfg = read_config()
#         spark = SparkSession.builder.getOrCreate()
#         cols = spark.table(cfg["input_table"]).columns
#         for lob in cfg.get("lobs", ["RES","COM","IND"]):
#             create_view_for_lob(spark, cfg, cols, lob)
#     except Exception as e:
#         print("ERROR:", e)
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()