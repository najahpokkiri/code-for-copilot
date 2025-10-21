#!/usr/bin/env python3
"""
Job5 - proportions x counts, verbose logging and robust fixes (v2)

Fixes included:
 - Detect combos that have NULL proportions (no matching (smod,built)) before replacing with zeros.
 - Replace NULL proportions with 0.0 for safe multiplication.
 - Create exp_* columns, then derive the actual exp column names dynamically from the DataFrame
   (prevents unresolved-column errors if existing intermediate tables have different names).
 - Write intermediate table using overwriteSchema to avoid schema-mismatch errors when re-running.
 - Export a small sample of missing combos to /dbfs/tmp/missing_props_sample.csv for inspection.
 - Better diagnostics: counts of combos, nulls, all-zero combos, and totals.

Usage (Databricks Spark Python Task):
Pass params as JSON array in the Task "Parameters" field, e.g.:
["--counts_table","prp_mr_bdap_projects.geospatialsolutions.counts_combined",
 "--proportions_table","prp_mr_bdap_projects.geospatialsolutions.proportions",
 "--out_table","prp_mr_bdap_projects.geospatialsolutions.estimates_combined",
 "--write_mode","overwrite",
 "--write_intermediate","True",
 "--output_dir","/dbfs/tmp/job5_debug",
 "--sample_limit","0"]

Before running:
 - If you want a clean intermediate write that replaces any old schema, keep write_intermediate=True.
 - If you prefer to inspect missing combos first, set write_intermediate=False and run; the script will still write the missing sample CSV.

Notes:
 - "NULL" in the joined proportions means the proportions table had no row for that (smod,built).
   The script reports how many combos lacked proportions and writes a small sample CSV so you can inspect and fix the proportions source.
"""

import os
import sys
import time
import traceback
import re
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType

# CLI wrapper for Databricks job JSON-params
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i].lstrip("-").upper()
            val = ""
            if i + 1 < len(args) and not args[i+1].startswith("--"):
                val = args[i+1]; i += 2
            else:
                i += 1
            if val != "":
                os.environ[key] = val
        else:
            i += 1

def getenv(k, default=None):
    v = os.environ.get(k.upper(), None)
    return default if v is None else v

# Parameters
COUNTS_TABLE = getenv("counts_table", None)
PROPS_TABLE = getenv("proportions_table", None)
OUT_TABLE = getenv("out_table", None)
WRITE_MODE = getenv("write_mode", "overwrite")
FLOOR_BINS_PARAM = getenv("floor_bins", "1,2,3,4-5,6-9,10-19,20+")
FLOOR_BINS = [s.strip() for s in FLOOR_BINS_PARAM.split(",") if s.strip()]
BIN_SUFFIXES = [re.sub(r'[^0-9A-Za-z]+', '_', b).strip('_') or "bin" for b in FLOOR_BINS]
SAMPLE_LIMIT = int(getenv("sample_limit", "0"))
WRITE_INTERMEDIATE = getenv("write_intermediate", "False").lower() in ("true","1","yes")
OUTPUT_DIR = getenv("output_dir", "/dbfs/tmp/job5_debug")
MAPPING_TABLE = getenv("mapping_table", None)

# Default built->sector mapping
DEFAULT_MAPPING = {
    11: "RES", 12: "RES", 13: "RES", 14: "RES", 15: "RES", 16: "RES",
    21: "COM", 22: "COM", 23: "COM",
    24: "IND", 25: "IND", 26: "IND"
}

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def print_sep(): print("-" * 80)

def main():
    start_all = time.time()
    spark = SparkSession.builder.getOrCreate()
    print("Job5 (fixed v2) start:", now())
    print("Params:", {"counts_table": COUNTS_TABLE, "proportions_table": PROPS_TABLE, "out_table": OUT_TABLE,
                      "floor_bins": FLOOR_BINS, "sample_limit": SAMPLE_LIMIT, "write_intermediate": WRITE_INTERMEDIATE,
                      "output_dir": OUTPUT_DIR})
    print_sep()

    if not COUNTS_TABLE or not PROPS_TABLE or not OUT_TABLE:
        raise RuntimeError("Provide --counts_table, --proportions_table and --out_table")

    # Read inputs
    if not spark.catalog.tableExists(COUNTS_TABLE):
        raise RuntimeError(f"Counts table not found: {COUNTS_TABLE}")
    if not spark.catalog.tableExists(PROPS_TABLE):
        raise RuntimeError(f"Proportions table not found: {PROPS_TABLE}")

    counts = spark.read.table(COUNTS_TABLE)
    props = spark.read.table(PROPS_TABLE)
    print("Read tables OK")
    print_sep()

    # Basic diagnostics
    print("Counts schema:")
    counts.printSchema()
    print("Proportions schema:")
    props.printSchema()
    print_sep()

    # normalize props column name if 'urban' exists
    if "smod" not in props.columns and "urban" in props.columns:
        props = props.withColumnRenamed("urban", "smod")
        print("Renamed 'urban' -> 'smod' in proportions DF")

    # check floor bins present
    missing_bins = [b for b in FLOOR_BINS if b not in props.columns]
    if missing_bins:
        raise RuntimeError(f"Proportions table missing expected bins: {missing_bins}")
    print("All floor bins present in proportions:", FLOOR_BINS)
    print_sep()

    # Optional sample mode
    if SAMPLE_LIMIT > 0:
        sample_ids = [r.grid_id for r in counts.select("grid_id").distinct().limit(SAMPLE_LIMIT).collect()]
        counts = counts.where(F.col("grid_id").isin(sample_ids))
        print(f"Running sample mode: limited counts to {counts.count()} rows")
        print_sep()

    # Detect built columns
    built_cols = [c for c in counts.columns if c.startswith("built_c_class_")]
    if not built_cols:
        raise RuntimeError("No built_c_class_* columns detected in counts table")
    built_codes = [int(c.replace("built_c_class_","")) for c in built_cols]
    print("Detected built columns:", built_cols)
    print_sep()

    # Build arrays and explode
    built_structs = [F.struct(F.lit(int(c.replace("built_c_class_",""))).alias("built"),
                              F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)).alias("built_count"))
                     for c in built_cols]
    df = counts.withColumn("_row_id", F.monotonically_increasing_id()).withColumn("_built_array", F.array(*built_structs))

    # Construct smod array
    if "smod_class" in counts.columns or "smod" in counts.columns:
        smod_col = "smod_class" if "smod_class" in counts.columns else "smod"
        df = df.withColumn("_smod_array", F.array(F.struct(F.col(smod_col).cast(IntegerType()).alias("smod"), F.lit(1.0).alias("smod_value"))))
        print("Using scalar smod column:", smod_col)
    else:
        smod_reclass_cols = [c for c in counts.columns if c.startswith("smod_reclass_")]
        if not smod_reclass_cols:
            raise RuntimeError("Counts table lacks smod_class and smod_reclass_*; cannot derive smod")
        smod_structs = [F.struct(F.lit(int(c.replace("smod_reclass_",""))).alias("smod"),
                                 F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)).alias("smod_value"))
                        for c in smod_reclass_cols]
        df = df.withColumn("_smod_array", F.array(*smod_structs))
        print("Using smod_reclass columns:", smod_reclass_cols)
    print_sep()

    df_built = df.select("_row_id","grid_id","tile_id", F.explode("_built_array").alias("b")) \
                 .select("_row_id","grid_id","tile_id", F.col("b.built").alias("built"), F.col("b.built_count").alias("built_count"))
    df_smod = df.select("_row_id", F.explode("_smod_array").alias("s")) \
                .select("_row_id", F.col("s.smod").alias("smod"), F.col("s.smod_value").alias("smod_value"))
    df_cross = df_built.join(df_smod, on="_row_id", how="inner")

    # weights
    smod_totals = df_smod.groupBy("_row_id").agg(F.sum("smod_value").alias("smod_total"))
    df_cross = df_cross.join(smod_totals, on="_row_id", how="left")
    df_cross = df_cross.withColumn("smod_weight", F.when(F.col("smod_total")==0, F.lit(0.0)).otherwise(F.col("smod_value")/F.col("smod_total")))
    df_cross = df_cross.withColumn("contribution", F.col("built_count") * F.col("smod_weight"))

    combo_counts = df_cross.groupBy("grid_id","tile_id","built","smod").agg(F.sum("contribution").alias("combo_count"))
    print("Computed combo_counts (sample):")
    combo_counts.limit(10).show(truncate=False)
    print_sep()

    # Prepare proportions table (coerce numeric)
    props_small = props.select("smod","built", *FLOOR_BINS).dropDuplicates(["smod","built"])
    for b in FLOOR_BINS:
        props_small = props_small.withColumn(b, F.col(b).cast(DoubleType()))

    # Join BEFORE replacing NULLs: detect combos with missing proportions
    combo_join_before = combo_counts.join(props_small, on=["smod","built"], how="left")
    # any null in any floor-bin means there was no matching proportions row (or proportion columns missing)
    any_null_cond = None
    for b in FLOOR_BINS:
        cond = F.col(b).isNull()
        any_null_cond = cond if any_null_cond is None else (any_null_cond | cond)
    missing_props_count = combo_join_before.filter(any_null_cond).count()
    total_combos = combo_join_before.count()
    print(f"Combos total={total_combos}; combos with NULL proportions (missing rows)={missing_props_count} ({100.0*missing_props_count/max(1,total_combos):.2f}%)")

    # write a small sample of missing combos for inspection
    if missing_props_count > 0:
        sample_missing = combo_join_before.filter(any_null_cond).select("grid_id","tile_id","built","smod", *FLOOR_BINS).limit(500)
        csv_out = OUTPUT_DIR.rstrip("/") + "/missing_props_sample"
        print(f"Writing sample of missing combos to {csv_out} (first 500 rows)")
        sample_missing.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_out)
        sample_missing.show(20, truncate=False)
    print_sep()

    # Now replace NULL proportions with 0.0 and proceed
    combo_with_props = combo_join_before
    for b in FLOOR_BINS:
        combo_with_props = combo_with_props.withColumn(b, F.coalesce(F.col(b), F.lit(0.0)))
    # Count combos where all proportions are zero (after replacement)
    zero_all_cond = None
    for b in FLOOR_BINS:
        eq = (F.col(b) == 0.0)
        zero_all_cond = eq if zero_all_cond is None else (zero_all_cond & eq)
    combos_all_zero = combo_with_props.filter(zero_all_cond).count()
    print(f"Combos total={total_combos}; combos where all proportions are zero={combos_all_zero} ({100.0*combos_all_zero/max(1,total_combos):.2f}%)")
    print_sep()

    # Compute exp columns
    for b, suf in zip(FLOOR_BINS, BIN_SUFFIXES):
        combo_with_props = combo_with_props.withColumn(f"exp_{suf}", F.col("combo_count") * F.col(b))

    # Build combo_expected
    # Select standard columns + dynamic exp columns (we'll discover actual names)
    combo_expected = combo_with_props.select("grid_id","tile_id","built","smod","combo_count", *[f"exp_{suf}" for suf in BIN_SUFFIXES])
    print("Combo_expected sample (first 10):")
    combo_expected.limit(10).show(truncate=False)

    # Determine actual exp column names (in case any previous runs left different names)
    actual_exp_cols = []
    for suf in BIN_SUFFIXES:
        candidates = [c for c in combo_expected.columns if c.startswith(f"exp_{suf}")]
        if not candidates:
            raise RuntimeError(f"Expected exp column for suffix {suf} not found in combo_expected columns: {combo_expected.columns}")
        actual_exp_cols.append(candidates[0])
    print("Actual exp columns:", actual_exp_cols)
    print_sep()

    # Optionally write intermediate combo_expected: overwrite schema to avoid mismatches with previous runs
    if WRITE_INTERMEDIATE:
        tmp_name = OUT_TABLE + "_combo_expected_debug"
        print("Writing intermediate combo_expected to:", tmp_name, "(overwriteSchema=True)")
        combo_expected.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(tmp_name)
        print("Wrote intermediate table.")
        print_sep()

    # Aggregate to grid_expected using the actual exp column names
    agg_exprs = [F.sum(F.col(c)).alias(f"expected_{suf}") for c, suf in zip(actual_exp_cols, BIN_SUFFIXES)]
    grid_expected = combo_expected.groupBy("grid_id","tile_id").agg(*agg_exprs)
    print("Grid_expected sample:")
    grid_expected.limit(10).show(truncate=False)
    print_sep()

    # Sector mapping and aggregation (default mapping used unless mapping table provided)
    if MAPPING_TABLE and spark.catalog.tableExists(MAPPING_TABLE):
        mapping_df = spark.read.table(MAPPING_TABLE).select(F.col("built").cast(IntegerType()).alias("built"), F.col("sector"))
        print("Loaded mapping table from", MAPPING_TABLE)
    else:
        mapping_df = spark.createDataFrame([(k,v) for k,v in DEFAULT_MAPPING.items()], schema=["built","sector"])
        print("Using default mapping (in-memory)")

    mapping_df.show(truncate=False)
    combo_with_sector = combo_expected.join(mapping_df, on="built", how="left")
    unmapped = combo_with_sector.filter(F.col("sector").isNull()).select("built").distinct().collect()
    if unmapped:
        print("WARNING: unmapped built codes:", [r.built for r in unmapped])
    combo_with_sector = combo_with_sector.fillna({"sector":"UNKNOWN"})

    sector_by_grid = combo_with_sector.groupBy("grid_id","tile_id","sector").agg(*[F.sum(F.col(c)).alias(c) for c in actual_exp_cols])
    print("sector_by_grid sample:")
    sector_by_grid.limit(20).show(truncate=False)
    print_sep()

    # Assemble final output: pivot sector_by_grid for each bin and join to grid_expected
    grid_meta = counts.select("grid_id","tile_id","centroid_x","centroid_y","lon","lat","i_idx","j_idx").dropDuplicates(["grid_id"])
    out_df = grid_expected.join(grid_meta, on=["grid_id","tile_id"], how="left")

    sectors = [r.sector for r in mapping_df.select("sector").distinct().collect()]
    if "UNKNOWN" in sectors:
        sectors = [s for s in sectors if s!="UNKNOWN"] + ["UNKNOWN"]
    print("Sectors:", sectors)

    for suf, exp_col in zip(BIN_SUFFIXES, actual_exp_cols):
        pivot = sector_by_grid.select("grid_id","sector", F.col(exp_col)).groupBy("grid_id").pivot("sector").sum(exp_col)
        for s in sectors:
            tgt = f"{s}_Storey_{suf}"
            if s in pivot.columns:
                pivot = pivot.withColumnRenamed(s, tgt)
            else:
                pivot = pivot.withColumn(tgt, F.lit(0.0))
        keep = ["grid_id"] + [f"{s}_Storey_{suf}" for s in sectors]
        pivot = pivot.select(*keep)
        out_df = out_df.join(pivot, on="grid_id", how="left")

    # Totals and percentages per sector
    for s in sectors:
        storey_cols = [f"{s}_Storey_{suf}" for suf in BIN_SUFFIXES]
        out_df = out_df.withColumn(f"{s}_Buildings_SUM", sum([F.coalesce(F.col(c), F.lit(0.0)) for c in storey_cols]))
        for suf in BIN_SUFFIXES:
            out_df = out_df.withColumn(f"{s}_Storey_{suf}_perc", F.when(F.col(f"{s}_Buildings_SUM")>0, F.col(f"{s}_Storey_{suf}")/F.col(f"{s}_Buildings_SUM")).otherwise(F.lit(0.0)))

    # Attach raw built sums and total_count
    built_sum_exprs = [F.sum(F.coalesce(F.col(f"built_c_class_{c}").cast(DoubleType()), F.lit(0.0))).alias(f"built_c_class_{c}") for c in built_codes]
    built_sums = counts.groupBy("grid_id").agg(*built_sum_exprs)
    out_df = out_df.join(built_sums, on="grid_id", how="left")
    out_df = out_df.withColumn("total_count", sum([F.coalesce(F.col(f"built_c_class_{c}"), F.lit(0.0)) for c in built_codes]))

    # TSI placeholders typed as Double
    for s in sectors:
        for suf in BIN_SUFFIXES:
            out_df = out_df.withColumn(f"{s}_Storey_{suf}_TSI", F.lit(None).cast(DoubleType()))
        out_df = out_df.withColumn(f"{s}_TSI_SUM", F.lit(None).cast(DoubleType()))

    # Fill numeric nulls -> 0 for clarity
    numeric_cols = [c for c,t in out_df.dtypes if t in ("double","int","bigint","long","float")]
    out_df = out_df.fillna({c: 0.0 for c in numeric_cols})

    # provenance
    out_df = out_df.withColumn("job_run_id", F.lit(f"job5_{int(time.time())}")).withColumn("run_ts", F.lit(time.strftime("%Y-%m-%d %H:%M:%S")))

    # Sanity checks: totals
    print("Sanity check: total expected sums by bin (sample):")
    expected_cols = [f"expected_{suf}" for suf in BIN_SUFFIXES]
    try:
        totals = out_df.select(*[F.sum(F.col(c)).alias(c) for c in expected_cols]).collect()[0].asDict()
        for k,v in totals.items():
            print(" ", k, "=", v)
    except Exception as e:
        print("Could not compute totals:", e)

    print("Raw built totals (per built class):")
    try:
        tot_built = counts.select(*[F.sum(F.col(c)).alias(c) for c in [f"built_c_class_{c}" for c in built_codes]]).collect()[0].asDict()
        for k,v in tot_built.items():
            print(" ", k, "=", v)
    except Exception as e:
        print("Could not compute raw built totals:", e)
    print_sep()

    # Write final table
    print("Writing final table:", OUT_TABLE)
    out_df.write.format("delta").mode(WRITE_MODE).option("overwriteSchema","true").saveAsTable(OUT_TABLE)
    print("Final table written.")
    print_sep()

    # Optionally write CSV snapshot
    if OUTPUT_DIR:
        out_csv = OUTPUT_DIR.rstrip("/") + "/final_counts_csv"
        print("Writing CSV snapshot to:", out_csv)
        out_df.coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv)
        print("CSV snapshot written.")
        print_sep()

    print("Job5 finished in %.2fs" % (time.time() - start_all))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Job5 failed:", e)
        traceback.print_exc()
        raise