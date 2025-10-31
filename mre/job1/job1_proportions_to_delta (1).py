
#!/usr/bin/env python3
"""
Task    1 — Load multipliers (proportions, TSI) into Delta tables

Converts a small notebook snippet into a runnable script that:
- loads config (JSON) from CONFIG_PATH (dbfs:/ or local) or uses env overrides,
- sets catalog & schema (optional),
- reads proportions CSV and tsi CSV (paths provided in config; supports dbfs:/ or local),
- renames column "story" -> "storey" if present in proportions,
- writes both DataFrames as managed Delta tables (table names from config),
- prints minimal diagnostics.

Usage:
 - Upload a config JSON to DBFS (e.g. dbfs:/configs/task5a_config.json).
 - Run as a Databricks job or on the driver with:
     python task5a_load_multipliers.py --config_path dbfs:/configs/task5a_config.json

Config keys (see example config file):
 - catalog (optional): catalog name to USE (e.g. prp_mr_bdap_projects)
 - schema  (optional): schema name to USE (e.g. geospatialsolutions)
 - proportions_csv_path: path to proportions CSV (can be dbfs:/... or local path)
 - tsi_csv_path: path to tsi CSV (can be dbfs:/... or local path)
 - proportions_table: fully-qualified target table name for proportions (catalog.schema.table)
 - tsi_table: fully-qualified target table name for tsi
 - csv_infer_schema: boolean (default true) - infer schema when reading CSV
 - csv_header: boolean (default true) - CSV has header
 - write_mode: "overwrite" or "append" (default "overwrite")
 - overwrite_schema: boolean (default true) - pass overwriteSchema when overwriting
 - preview: boolean (default true) - show first few rows after read
"""



# os.environ["CONFIG_PATH"] = "./config.json"
import os
import sys
import json
import time
import traceback
from typing import Dict, Any

import pandas as pd
from pyspark.sql import SparkSession
os.environ["CONFIG_PATH"] = "./config.json"
DEFAULT_CONFIG: Dict[str, Any] = {
    "catalog": "prp_mr_bdap_projects",
    "schema": "geospatialsolutions",
    "proportions_csv_path": None,
    "tsi_csv_path": None,
    "proportions_table": "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_proportions_input",
    "tsi_table": "prp_mr_bdap_projects.geospatialsolutions.building_enrichment_tsi_input",
    "csv_infer_schema": True,
    "csv_header": True,
    "write_mode": "overwrite",
    "overwrite_schema": True,
    "preview": True,
    "preview_rows": 5
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
        try:
            loaded = _read_json_path(cfg_path)
            if not isinstance(loaded, dict):
                raise ValueError("config file must be a JSON object")
            cfg.update(loaded)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{cfg_path}': {e}")

    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            val = os.environ[env_key]
            if isinstance(cfg[key], bool):
                cfg[key] = str(val).lower() in ("true","1","t","yes")
            else:
                cfg[key] = val
    missing = [k for k in ("proportions_csv_path", "tsi_csv_path") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {missing}. Add them to config or pass overrides.")
    return cfg

def resolve_path(path: str) -> str:
    if isinstance(path, str) and path.startswith("dbfs:"):
        return path.replace("dbfs:", "/dbfs", 1)
    return path

def main():
    start = time.time()
    cfg = load_config()
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

    print("Task5a config (effective):")
    for k in ["catalog","schema","proportions_csv_path","tsi_csv_path","proportions_table","tsi_table","write_mode","overwrite_schema","preview"]:
        print(f" {k:22} = {cfg.get(k)}")

    spark = SparkSession.builder.getOrCreate()

    catalog = cfg.get("catalog")
    schema = cfg.get("schema")

    if catalog:
        try:
            spark.sql(f"USE CATALOG {catalog}")
            print(f"Set catalog to: {catalog}")
        except Exception as e:
            print(f"Warning: failed to USE CATALOG {catalog}: {e}")

    if schema:
        try:
            spark.sql(f"USE SCHEMA {schema}")
            print(f"Set schema to: {schema}")
        except Exception as e:
            try:
                spark.sql(f"USE {catalog}.{schema}")
                print(f"Fallback: used legacy namespace: {catalog}.{schema}")
            except Exception as e2:
                print(f"Warning: failed to USE SCHEMA {catalog}.{schema}: {e2}")

    # Use pandas for fast small CSV read
    prop_path = resolve_path(cfg["proportions_csv_path"])
    tsi_path = resolve_path(cfg["tsi_csv_path"])

    print(f"Reading proportions CSV from: {prop_path}")
    prop_pd = pd.read_csv(prop_path)
    if "story" in prop_pd.columns:
        prop_pd = prop_pd.rename(columns={"story": "storey"})
    elif "storey" not in prop_pd.columns:
        raise RuntimeError("proportions CSV must contain 'story' or 'storey' column")
    prop_df = spark.createDataFrame(prop_pd)

    if cfg.get("preview", True):
        print("Proportions sample:")
        prop_df.show(cfg.get("preview_rows", 5), truncate=False)
        prop_df.printSchema()

    print(f"Reading TSI CSV from: {tsi_path}")
    tsi_pd = pd.read_csv(tsi_path)
    tsi_df = spark.createDataFrame(tsi_pd)
    if cfg.get("preview", True):
        print("TSI sample:")
        tsi_df.show(cfg.get("preview_rows", 5), truncate=False)
        tsi_df.printSchema()

    # Suffix output table names only
    proportions_table = add_iso_suffix(cfg.get("proportions_table"))
    tsi_table = add_iso_suffix(cfg.get("tsi_table"))
    write_mode = cfg.get("write_mode", "overwrite")
    overwrite_schema = bool(cfg.get("overwrite_schema", True))

    try:
        print(f"Writing proportions to table {proportions_table} (mode={write_mode})")
        writer = prop_df.write.format("delta").mode(write_mode)
        if write_mode == "overwrite" and overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(proportions_table)
        print("Proportions table write complete.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR writing proportions table: {e}\n{tb}")
        raise

    try:
        print(f"Writing TSI to table {tsi_table} (mode={write_mode})")
        writer = tsi_df.write.format("delta").mode(write_mode)
        if write_mode == "overwrite" and overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(tsi_table)
        print("TSI table write complete.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERROR writing TSI table: {e}\n{tb}")
        raise

    try:
        print("Available tables in schema:")
        spark.sql(f"SHOW TABLES IN {catalog}.{schema}").show(truncate=False)
    except Exception:
        spark.sql("SHOW TABLES").show(truncate=False)

    elapsed = time.time() - start
    print(f"Task5a completed in {elapsed:.1f}s")

if __name__ == "__main__":
    main()