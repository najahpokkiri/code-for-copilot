#!/usr/bin/env python3
"""
Task 1 — Load Multipliers (Proportions & TSI) into Delta Tables

Loads building storey proportions and Total Sum Insured (TSI) multipliers from
CSV files into Delta tables for use by downstream tasks.

Configuration:
--------------
Reads from config.json (generated from config.yaml via config_builder.py).
All table names are auto-generated from the YAML configuration.

Required config keys:
  - proportions_csv_path: Path to proportions CSV
  - tsi_csv_path: Path to TSI multipliers CSV
  - proportions_table: Output Delta table for proportions
  - tsi_table: Output Delta table for TSI
  - iso3: Country ISO3 code
  - catalog: Databricks catalog
  - schema: Databricks schema

Usage:
------
  python task1_proportions_to_delta.py --config_path config.json

Or with CLI overrides:
  python task1_proportions_to_delta.py --config_path config.json --iso3 USA

Output:
-------
  - Delta table: {catalog}.{schema}.inv_NoS_{ISO3}_storey_mapping
  - Delta table: {catalog}.{schema}.inv_NoS_{ISO3}_tsi

ROBUST FEATURES:
================
1. **Intelligent Numeric Format Detection**:
   - Handles comma decimals (European: 0,49 → 0.49)
   - Handles period decimals (US: 0.49 → 0.49)
   - Handles integer percentages (60 → 0.60)
   - Handles percentage strings (60% → 0.60)
   - Handles mixed formats within same file

2. **Smart Sum Detection & Normalization**:
   - Auto-detects if values sum to ~100 (percentages) or ~1.0 (decimals)
   - Normalizes to decimals (0.0-1.0) automatically
   - Validates row sums and warns about anomalies
   - Optional auto-normalization of invalid rows

3. **Robust Column Mapping**:
   - Maps various bin headers (4-5, 4_5, 9-20, 10-20, 20+, etc.)
   - Handles 'built' or 'storey' column names
   - Drops trailing SUM/total columns automatically

4. **Fallback Handling**:
   - Multiple separator detection (comma, semicolon)
   - Encoding detection (UTF-8, Latin-1)
   - Missing column handling with defaults

Proportions Format:
-------------------
Expected columns: built, 1, 2, 3, 4_5, 6_8, 9_20, 20
Values should sum to 1.0 (or 100% if percentages)

TSI Format:
-----------
Expected columns: built, tsi_m2
"""

import os
import sys
import json
import time
import traceback
import re
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
# os.environ["CONFIG_PATH"] = "./config.json"
# ================================================================================
# CLI ARGUMENT PARSER (Maps --key value to UPPERCASE env vars)
# ================================================================================
if len(sys.argv) > 1:
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            k = a.lstrip("-").upper()
            v = ""
            if (i + 1) < len(args) and not args[i + 1].startswith("--"):
                v = args[i + 1]
                i += 2
            else:
                i += 1
            if v != "":
                os.environ[k] = v
        else:
            i += 1

# ================================================================================
# DEFAULT CONFIG
# ================================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "catalog": "prp_mr_bdap_projects",
    "schema": "geospatialsolutions",
    "proportions_csv_path": None,
    "tsi_csv_path": None,
    "proportions_table": None,
    "tsi_table":None,
    "iso3": "IND",
    "csv_infer_schema": True,
    "csv_header": True,
    "write_mode": "overwrite",
    "overwrite_schema": True,
    "preview": True,
    "preview_rows": 5,
    "auto_normalize_bad_rows": False,  # Set True to force normalization
    "write_audit": True,
    "sum_tolerance": 0.02  # Allow ±2% deviation from 1.0
}

# Canonical storey bin names (used by downstream tasks)
CANON_BINS = ["1", "2", "3", "4_5", "6_8", "9_20", "20"]

# Flexible bin header mapping (handles common variations)
INCOMING_BIN_MAP = {
    "1": "1", "01": "1",
    "2": "2", "02": "2",
    "3": "3", "03": "3",
    "4-5": "4_5", "4_5": "4_5", "4-5 ": "4_5", "4_5 ": "4_5",
    "6-8": "6_8", "6_8": "6_8", "6-9": "6_8", "6_9": "6_8",
    "9-20": "9_20", "9_20": "9_20", "10-20": "9_20", "10_20": "9_20",
    "20+": "20", "20": "20", "20 ": "20", ">20": "20"
}

# ================================================================================
# CONFIG LOADER
# ================================================================================
def _read_json_path(path: str) -> Dict[str, Any]:
    """Read JSON from dbfs:/ or local path"""
    if path.startswith("dbfs:"):
        local_path = path.replace("dbfs:", "/dbfs", 1)
    else:
        local_path = path
    with open(local_path, "r", encoding='utf-8') as f:
        return json.load(f)

def load_config() -> Dict[str, Any]:
    """Load config from file + env overrides"""
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = os.environ.get("CONFIG_PATH") or os.environ.get("CONFIG") or os.environ.get("CONFIG_JSON", "")
    
    if cfg_path:
        try:
            loaded = _read_json_path(cfg_path)
            if not isinstance(loaded, dict):
                raise ValueError("config file must be a JSON object")
            cfg.update(loaded)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{cfg_path}': {e}")
    
    # Apply env overrides
    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ and os.environ[env_key] != "":
            val = os.environ[env_key]
            if isinstance(cfg[key], bool):
                cfg[key] = str(val).lower() in ("true", "1", "t", "yes")
            elif isinstance(cfg[key], (int, float)):
                try:
                    cfg[key] = type(cfg[key])(val)
                except:
                    pass
            else:
                cfg[key] = val
    
    # Validate required keys
    missing = [k for k in ("proportions_csv_path", "tsi_csv_path") if not cfg.get(k)]
    if missing:
        raise RuntimeError(f"Missing required config keys: {missing}")
    
    return cfg

def resolve_path(path: str) -> str:
    """Convert dbfs:/ to /dbfs/ for local access"""
    if isinstance(path, str) and path.startswith("dbfs:"):
        return path.replace("dbfs:", "/dbfs", 1)
    return path

def add_iso_suffix(name: Optional[str], iso3: str) -> Optional[str]:
    """Add ISO3 suffix if not already present"""
    if name is None:
        return name
    if name.upper().endswith(f"_{iso3}"):
        return name
    if name.endswith(".csv"):
        return name[:-4] + f"_{iso3}.csv"
    if name.endswith(".parquet"):
        return name[:-8] + f"_{iso3}.parquet"
    return f"{name}_{iso3}"

# ================================================================================
# INTELLIGENT NUMERIC PARSER
# ================================================================================
def parse_numeric_value(value: Any) -> float:
    """
    Parse ANY numeric format into a float:
    - Handles comma decimals: "0,49" → 0.49
    - Handles percentages: "60%", "60" → 0.60 (if detected as percentage)
    - Handles already-decimal: "0.49" → 0.49
    - Returns NaN for unparseable values
    """
    if pd.isna(value):
        return np.nan
    
    # Convert to string and clean
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    
    # Remove whitespace
    s = s.replace(" ", "")
    
    # Check for percentage sign
    has_percent = "%" in s
    s = s.replace("%", "")
    
    # Replace comma with period for decimal parsing
    s = s.replace(",", ".")
    
    try:
        v = float(s)
    except:
        return np.nan
    
    # If had % sign, definitely convert from percentage
    if has_percent:
        return v / 100.0
    
    # Return as-is (caller will determine if it needs percentage conversion)
    return v

def detect_value_scale(series: pd.Series) -> Tuple[str, float]:
    """
    Detect if column contains decimals (0-1) or percentages (0-100+).
    
    Returns:
        ("decimal", median) or ("percentage", median)
    """
    # Parse all values
    parsed = series.apply(parse_numeric_value).dropna()
    
    if len(parsed) == 0:
        return ("decimal", 0.0)
    
    median = float(parsed.median())
    max_val = float(parsed.max())
    min_val = float(parsed.min())
    
    # Decision logic:
    # - If median > 1.5 → percentage
    # - If max > 10 → percentage
    # - If all values < 1.1 → decimal
    
    if median > 1.5 or max_val > 10:
        return ("percentage", median)
    elif max_val <= 1.1 and min_val >= 0.0:
        return ("decimal", median)
    else:
        # Mixed or ambiguous: use median to decide
        return ("percentage" if median > 1.0 else "decimal", median)

def detect_row_scale(row_values: pd.Series) -> str:
    """
    Detect if a ROW contains percentages or decimals.
    Used for row-by-row normalization.
    """
    total = row_values.sum()
    if pd.isna(total):
        return "unknown"
    
    # If sum is close to 1.0 (±0.2), it's decimal
    if 0.8 <= total <= 1.2:
        return "decimal"
    # If sum is close to 100 (±20), it's percentage
    elif 80 <= total <= 120:
        return "percentage"
    else:
        # Ambiguous - use median value to decide
        median = row_values.median()
        return "percentage" if median > 1.0 else "decimal"

# ================================================================================
# CSV DETECTION HELPERS
# ================================================================================
def detect_separator(path: str) -> str:
    """Detect CSV separator (comma vs semicolon)"""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            sample = fh.read(4096)
    except:
        try:
            with open(path, 'r', encoding='latin-1') as fh:
                sample = fh.read(4096)
        except:
            return ','
    
    # Count occurrences
    comma_count = sample.count(',')
    semicolon_count = sample.count(';')
    tab_count = sample.count('\t')
    
    if semicolon_count > comma_count and semicolon_count > tab_count:
        return ';'
    elif tab_count > comma_count and tab_count > semicolon_count:
        return '\t'
    else:
        return ','

def detect_encoding(path: str) -> str:
    """Detect file encoding"""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            fh.read(1024)
        return 'utf-8'
    except:
        return 'latin-1'

# ================================================================================
# COLUMN MAPPING
# ================================================================================
def map_bin_columns(columns: list) -> list:
    """
    Map incoming column headers to canonical bin names.
    Returns: [(incoming_col, canonical_name), ...]
    """
    mapped = []
    
    for col in columns:
        col_clean = col.strip()
        col_norm = col_clean.replace(' ', '').replace('-', '_').replace('+', '').lower()
        
        # Try exact match first
        if col_clean in CANON_BINS:
            mapped.append((col, col_clean))
            continue
        
        # Try mapping table
        matched = False
        for incoming_pattern, canonical in INCOMING_BIN_MAP.items():
            if (col_clean.lower() == incoming_pattern.lower() or 
                col_norm == incoming_pattern.replace('-', '_').lower()):
                mapped.append((col, canonical))
                matched = True
                break
        
        if not matched:
            # Try regex patterns for bins like "9-20", "10-20", etc.
            if re.match(r'^\d+[-_]\d+$', col_clean):
                # Extract numbers
                parts = re.split(r'[-_]', col_clean)
                if len(parts) == 2:
                    low, high = parts
                    # Map to closest canonical bin
                    if high in ('5', '05'):
                        mapped.append((col, '4_5'))
                    elif high in ('8', '9', '08', '09'):
                        mapped.append((col, '6_8'))
                    elif int(high) >= 10:
                        mapped.append((col, '9_20'))
    
    return mapped

# ================================================================================
# MAIN PROPORTIONS NORMALIZER
# ================================================================================
def normalize_proportions_dataframe(
    prop_df: pd.DataFrame,
    cfg: Dict[str, Any]
) -> pd.DataFrame:
    """
    Intelligent normalization of proportions CSV to canonical format.
    
    Handles:
    - Comma/period decimals
    - Integer percentages vs decimal proportions
    - Column mapping to canonical bins
    - Row sum validation and normalization
    """
    
    print("\n" + "="*80)
    print("PROPORTIONS NORMALIZATION")
    print("="*80)
    
    # Clean column names
    prop_df.columns = [c.strip() for c in prop_df.columns]
    print(f"\nOriginal columns: {list(prop_df.columns)}")
    print(f"Original shape: {prop_df.shape}")
    
    # -------------------------------------------------------------------------
    # 1. DROP TRAILING SUM/TOTAL COLUMN
    # -------------------------------------------------------------------------
    if len(prop_df.columns) > 0:
        last_col = prop_df.columns[-1]
        if last_col.upper() in ('SUM', 'TOTAL'):
            print(f"\n✓ Dropping trailing SUM column: {last_col}")
            prop_df = prop_df.drop(columns=[last_col])
    
    # -------------------------------------------------------------------------
    # 2. NORMALIZE METADATA COLUMNS
    # -------------------------------------------------------------------------
    # Handle 'built' or 'storey'
    if 'built' in prop_df.columns and 'storey' not in prop_df.columns:
        prop_df = prop_df.rename(columns={'built': 'storey'})
    elif 'storey' not in prop_df.columns and 'built' not in prop_df.columns:
        raise RuntimeError("CSV must contain 'built' or 'storey' column")
    
    # Ensure LOB column exists
    if 'lob' in prop_df.columns and 'LOB' not in prop_df.columns:
        prop_df['LOB'] = prop_df['lob']
    if 'LOB' not in prop_df.columns:
        # Try to infer from storey ranges
        print("⚠ Warning: No LOB column found, will attempt to infer from storey values")
    
    # -------------------------------------------------------------------------
    # 3. MAP BIN COLUMNS
    # -------------------------------------------------------------------------
    mapped_cols = map_bin_columns(prop_df.columns)
    
    if not mapped_cols:
        # Fallback: try exact canonical matches
        for b in CANON_BINS:
            if b in prop_df.columns:
                mapped_cols.append((b, b))
    
    if not mapped_cols:
        raise RuntimeError(
            f"No bin columns detected!\n"
            f"Columns found: {list(prop_df.columns)}\n"
            f"Expected patterns: {list(INCOMING_BIN_MAP.keys())}"
        )
    
    print(f"\n✓ Mapped {len(mapped_cols)} bin columns:")
    for incoming, canonical in mapped_cols:
        print(f"  {incoming:15} → {canonical}")
    
    # -------------------------------------------------------------------------
    # 4. BUILD OUTPUT DATAFRAME WITH METADATA
    # -------------------------------------------------------------------------
    out = pd.DataFrame()
    
    # Copy metadata columns
    for col in ['ISO3', 'LOB', 'urban', 'storey']:
        if col in prop_df.columns:
            out[col] = prop_df[col].astype(str).str.strip()
    
    # Ensure ISO3 exists
    if 'ISO3' not in out.columns or out['ISO3'].isnull().all():
        iso3 = cfg.get('iso3', 'IND').upper()
        out['ISO3'] = iso3
        print(f"\n✓ Added ISO3 column: {iso3}")
    
    # -------------------------------------------------------------------------
    # 5. PARSE AND CONVERT NUMERIC COLUMNS
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("NUMERIC CONVERSION & SCALE DETECTION")
    print("="*80)
    
    # First pass: parse all values
    parsed_cols = {}
    for incoming, canonical in mapped_cols:
        parsed_cols[canonical] = prop_df[incoming].apply(parse_numeric_value)
    
    # Detect scale for each column
    print("\nColumn-wise scale detection:")
    column_scales = {}
    for canonical, parsed_series in parsed_cols.items():
        scale, median = detect_value_scale(parsed_series)
        column_scales[canonical] = scale
        print(f"  {canonical:6} : {scale:10} (median={median:.2f})")
    
    # Detect overall file scale (majority vote)
    scale_counts = pd.Series(list(column_scales.values())).value_counts()
    overall_scale = scale_counts.idxmax() if len(scale_counts) > 0 else "decimal"
    
    print(f"\n✓ Overall file scale detected: {overall_scale.upper()}")
    
    # Convert to decimals if needed
    for canonical, parsed_series in parsed_cols.items():
        col_scale = column_scales[canonical]
        
        if col_scale == "percentage":
            # Convert percentage to decimal
            out[canonical] = (parsed_series / 100.0).fillna(0.0)
        else:
            # Already decimal
            out[canonical] = parsed_series.fillna(0.0)
    
    # Ensure all canonical bins exist
    for b in CANON_BINS:
        if b not in out.columns:
            out[b] = 0.0
            print(f"  ⚠ Added missing bin column: {b} (filled with 0.0)")
    
    # -------------------------------------------------------------------------
    # 6. VALIDATE AND NORMALIZE ROW SUMS
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("ROW SUM VALIDATION")
    print("="*80)
    
    # Compute row sums
    out['_row_sum'] = out[CANON_BINS].sum(axis=1)
    
    sum_min = out['_row_sum'].min()
    sum_max = out['_row_sum'].max()
    sum_mean = out['_row_sum'].mean()
    sum_median = out['_row_sum'].median()
    
    print(f"\nRow sum statistics:")
    print(f"  Min    : {sum_min:.4f}")
    print(f"  Max    : {sum_max:.4f}")
    print(f"  Mean   : {sum_mean:.4f}")
    print(f"  Median : {sum_median:.4f}")
    
    # Check tolerance
    tolerance = float(cfg.get('sum_tolerance', 0.02))
    target = 1.0
    
    bad_rows = out[
        (out['_row_sum'] < target - tolerance) | 
        (out['_row_sum'] > target + tolerance)
    ]
    
    if len(bad_rows) > 0:
        print(f"\n⚠ WARNING: {len(bad_rows)}/{len(out)} rows outside tolerance (±{tolerance*100:.0f}%)")
        print(f"\nSample problematic rows:")
        display_cols = ['storey', 'LOB', 'urban', '_row_sum'] + CANON_BINS
        display_cols = [c for c in display_cols if c in bad_rows.columns]
        print(bad_rows[display_cols].head(10).to_string(index=False))
        
        # Auto-normalize if requested
        if cfg.get('auto_normalize_bad_rows', False):
            print(f"\n✓ Auto-normalization ENABLED - rescaling bad rows...")
            
            # Only normalize rows with non-zero sums
            idx_bad = bad_rows.index
            non_zero_mask = out.loc[idx_bad, '_row_sum'] > 0.001
            idx_to_fix = idx_bad[non_zero_mask]
            
            if len(idx_to_fix) > 0:
                # Rescale each bin proportionally
                for bin_col in CANON_BINS:
                    out.loc[idx_to_fix, bin_col] = (
                        out.loc[idx_to_fix, bin_col] / out.loc[idx_to_fix, '_row_sum']
                    )
                
                # Recompute sums
                out['_row_sum'] = out[CANON_BINS].sum(axis=1)
                
                print(f"  After normalization:")
                print(f"    Min    : {out['_row_sum'].min():.4f}")
                print(f"    Max    : {out['_row_sum'].max():.4f}")
                print(f"    Mean   : {out['_row_sum'].mean():.4f}")
        else:
            print(f"\n  ℹ Auto-normalization DISABLED (set auto_normalize_bad_rows=True to enable)")
    else:
        print(f"\n✓ All rows sum to ~{target} within tolerance")
    
    # Drop temporary sum column
    out = out.drop(columns=['_row_sum'])
    
    # -------------------------------------------------------------------------
    # 7. ENSURE PROPER DATA TYPES
    # -------------------------------------------------------------------------
    out['storey'] = pd.to_numeric(out['storey'], errors='coerce').astype('Int64')
    out['urban'] = pd.to_numeric(out['urban'], errors='coerce').fillna(0).astype(int)
    out['LOB'] = out['LOB'].astype(str)
    out['ISO3'] = out['ISO3'].astype(str)
    
    for b in CANON_BINS:
        out[b] = out[b].astype(float)
    
    print(f"\n{'='*80}")
    print(f"✓ NORMALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Output shape: {out.shape}")
    print(f"  Final columns: {list(out.columns)}")
    
    return out

# ================================================================================
# MAIN FUNCTION
# ================================================================================
def main():
    start_time = time.time()
    
    print("\n" + "="*80)
    print("TASK 1: LOAD PROPORTIONS & TSI TO DELTA TABLES (ROBUST VERSION)")
    print("="*80 + "\n")
    
    # Load config
    cfg = load_config()
    ISO3 = cfg.get("iso3", "IND").strip().upper()
    
    print("Configuration:")
    print("-" * 80)
    for k in ["catalog", "schema", "proportions_csv_path", "tsi_csv_path", 
              "proportions_table", "tsi_table", "iso3", "write_mode", 
              "auto_normalize_bad_rows", "write_audit"]:
        print(f"  {k:28} : {cfg.get(k)}")
    print()
    
    # Initialize Spark
    spark = SparkSession.builder.getOrCreate()
    
    # Set catalog/schema
    catalog = cfg.get("catalog")
    schema = cfg.get("schema")
    
    if catalog:
        try:
            spark.sql(f"USE CATALOG {catalog}")
            print(f"✓ Using catalog: {catalog}")
        except Exception as e:
            print(f"⚠ Warning: Could not set catalog {catalog}: {e}")
    
    if schema:
        try:
            spark.sql(f"USE SCHEMA {schema}")
            print(f"✓ Using schema: {schema}")
        except Exception as e:
            try:
                spark.sql(f"USE {catalog}.{schema}")
                print(f"✓ Using namespace: {catalog}.{schema}")
            except Exception as e2:
                print(f"⚠ Warning: Could not set schema: {e2}")
    
    # =========================================================================
    # PROCESS PROPORTIONS CSV
    # =========================================================================
    prop_path = resolve_path(cfg["proportions_csv_path"])
    
    print(f"\n{'='*80}")
    print("LOADING PROPORTIONS CSV")
    print(f"{'='*80}")
    print(f"Path: {prop_path}")
    
    # Detect file properties
    separator = detect_separator(prop_path)
    encoding = detect_encoding(prop_path)
    
    print(f"Detected separator: '{separator}'")
    print(f"Detected encoding: {encoding}")
    
    # Read CSV
    try:
        prop_df_raw = pd.read_csv(
            prop_path,
            sep=separator,
            encoding=encoding,
            dtype=str  # Read as string first for robust parsing
        )
        print(f"✓ Loaded {len(prop_df_raw)} rows, {len(prop_df_raw.columns)} columns")
    except Exception as e:
        print(f"✗ ERROR loading CSV: {e}")
        raise
    
    # Normalize proportions
    try:
        prop_df_normalized = normalize_proportions_dataframe(prop_df_raw, cfg)
    except Exception as e:
        print(f"\n✗ ERROR during normalization:")
        traceback.print_exc()
        raise
    
    # Convert to Spark DataFrame
    prop_spark_df = spark.createDataFrame(prop_df_normalized)
    
    if cfg.get("preview", True):
        print(f"\n{'='*80}")
        print(f"PROPORTIONS PREVIEW (first {cfg.get('preview_rows', 5)} rows)")
        print(f"{'='*80}")
        prop_spark_df.show(cfg.get('preview_rows', 5), truncate=False)
        print("\nSchema:")
        prop_spark_df.printSchema()
    
    # Write to Delta
    proportions_table = add_iso_suffix(cfg.get("proportions_table"), ISO3)
    write_mode = cfg.get("write_mode", "overwrite")
    overwrite_schema = bool(cfg.get("overwrite_schema", True))
    
    print(f"\n{'='*80}")
    print("WRITING PROPORTIONS TO DELTA")
    print(f"{'='*80}")
    print(f"Table: {proportions_table}")
    print(f"Mode: {write_mode}")
    
    try:
        writer = prop_spark_df.write.format("delta").mode(write_mode)
        if write_mode == "overwrite" and overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(proportions_table)
        print(f"✓ Proportions table written successfully")
    except Exception as e:
        print(f"✗ ERROR writing proportions table:")
        traceback.print_exc()
        raise
    
    # Write audit table if requested
    if cfg.get('write_audit', True):
        try:
            audit_table = proportions_table + "_audit"
            audit_df = prop_df_normalized.copy()
            audit_df['_row_sum'] = audit_df[CANON_BINS].sum(axis=1)
            audit_spark_df = spark.createDataFrame(audit_df)
            
            audit_spark_df.write.format("delta").mode("overwrite") \
                .option("overwriteSchema", "true").saveAsTable(audit_table)
            
            print(f"✓ Wrote audit table: {audit_table}")
        except Exception as e:
            print(f"⚠ Warning: Could not write audit table: {e}")
    
    # =========================================================================
    # PROCESS TSI CSV (Simple pass-through)
    # =========================================================================
    tsi_path = resolve_path(cfg["tsi_csv_path"])
    
    print(f"\n{'='*80}")
    print("LOADING TSI CSV")
    print(f"{'='*80}")
    print(f"Path: {tsi_path}")
    
    try:
        tsi_df = pd.read_csv(tsi_path)
        print(f"✓ Loaded {len(tsi_df)} rows, {len(tsi_df.columns)} columns")
        
        tsi_spark_df = spark.createDataFrame(tsi_df)
        
        if cfg.get("preview", True):
            print(f"\nTSI Preview:")
            tsi_spark_df.show(cfg.get('preview_rows', 5), truncate=False)
        
        tsi_table = add_iso_suffix(cfg.get("tsi_table"), ISO3)
        
        print(f"\nWriting to table: {tsi_table}")
        writer = tsi_spark_df.write.format("delta").mode(write_mode)
        if write_mode == "overwrite" and overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(tsi_table)
        
        print(f"✓ TSI table written successfully")
        
    except Exception as e:
        print(f"✗ ERROR processing TSI:")
        traceback.print_exc()
        raise
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print("TABLES IN SCHEMA")
    print(f"{'='*80}")
    try:
        spark.sql(f"SHOW TABLES IN {catalog}.{schema}").show(truncate=False)
    except:
        spark.sql("SHOW TABLES").show(truncate=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✓ TASK 1 COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Proportions table: {proportions_table}")
    print(f"  TSI table: {tsi_table}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

