#!/usr/bin/env python3
"""
Job1 - Ingest absolute-floor-count CSV -> Delta (counts + proportions).
Minimal, safe: backs up existing proportions table (if present) and overwrites schema.

Usage: pass parameters as a Databricks Spark Python Task Parameters JSON array, e.g.
["--input_csv","/dbfs/FileStore/proportions_input.csv",
 "--counts_table","prp_mr_bdap_projects.geospatialsolutions.proportions_counts",
 "--proportions_table","prp_mr_bdap_projects.geospatialsolutions.proportions",
 "--write_mode","overwrite"]

Parameters:
 - input_csv (required)      : CSV path (DBFS or mounted path)
 - counts_table (required)   : Delta table name for absolute counts
 - proportions_table (req)   : Delta table name for normalized proportions
 - floor_bins (optional)     : comma-separated bin column names, default "1,2,3,4-5,6-9,10-19,20+"
 - backup (optional)         : "true"/"false" default "true" — create backup before overwrite
 - write_mode (optional)     : default "overwrite"
"""
import os, sys, time, traceback
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType

# CLI wrapper: turn JSON-style args into env vars (Databricks Task)
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            k = args[i].lstrip("-").upper()
            v = ""
            if i + 1 < len(args) and not args[i+1].startswith("--"):
                v = args[i+1]; i += 2
            else:
                i += 1
            if v != "":
                os.environ[k] = v
        else:
            i += 1

def getenv(k, default=None):
    v = os.environ.get(k.upper(), None)
    return default if v is None else v

def main():
    spark = SparkSession.builder.getOrCreate()
    input_csv = getenv("input_csv")
    counts_table = getenv("counts_table")
    props_table = getenv("proportions_table")
    floor_bins = [b.strip() for b in getenv("floor_bins","1,2,3,4-5,6-9,10-19,20+").split(",")]
    do_backup = getenv("backup","true").lower() in ("true","1","yes")
    write_mode = getenv("write_mode","overwrite")

    if not input_csv or not counts_table or not props_table:
        raise RuntimeError("Missing required params: --input_csv, --counts_table, --proportions_table")

    print("Job1 ingest start", input_csv, counts_table, props_table)
    df = spark.read.option("header", True).option("inferSchema", True).csv(input_csv)
    print("CSV schema (inferred):"); df.printSchema()

    # normalize name
    if "urban" in df.columns and "smod" not in df.columns:
        df = df.withColumnRenamed("urban","smod")
        print("Renamed 'urban' -> 'smod'")

    # validate bins
    missing = [b for b in floor_bins if b not in df.columns]
    if missing:
        raise RuntimeError(f"Missing floor-bin columns in CSV: {missing}")

    # cast columns
    df = df.withColumn("smod", F.col("smod").cast(IntegerType()))
    df = df.withColumn("built", F.col("built").cast(IntegerType()))
    for b in floor_bins:
        df = df.withColumn(b, F.coalesce(F.col(b).cast(DoubleType()), F.lit(0.0)))

    # add provenance
    df = df.withColumn("ingested_at", F.lit(time.strftime("%Y-%m-%d %H:%M:%S")))
    df = df.withColumn("source_file", F.lit(input_csv))

    # backup existing proportions table if requested
    if do_backup and spark.catalog.tableExists(props_table):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_name = props_table + "_backup_" + ts
        print("Backing up existing proportions table to:", backup_name)
        spark.sql(f"CREATE TABLE {backup_name} AS SELECT * FROM {props_table}")
        print("Backup created.")

    # write counts_table (raw absolute)
    print("Writing counts table:", counts_table)
    df.write.format("delta").option("overwriteSchema","true").mode(write_mode).saveAsTable(counts_table)
    print("Counts table written.")

    # build normalized proportions per row
    sum_expr = None
    for b in floor_bins:
        sum_expr = F.col(b) if sum_expr is None else sum_expr + F.col(b)
    props = df.withColumn("_row_total", sum_expr)
    for b in floor_bins:
        props = props.withColumn(b, F.when(F.col("_row_total") > 0, F.col(b) / F.col("_row_total")).otherwise(F.lit(0.0)))
    select_cols = ["smod","built"] + floor_bins + ["ingested_at","source_file"]
    props_out = props.select(*select_cols)

    # write normalized proportions (overwrite schema to match CSV canonical)
    print("Writing proportions table:", props_table)
    props_out.write.format("delta").option("overwriteSchema","true").mode(write_mode).saveAsTable(props_table)
    print("Proportions table written.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Job1 failed:", e)
        traceback.print_exc()
        raise