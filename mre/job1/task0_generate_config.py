#!/usr/bin/env python3
"""
Task 0 — Generate Configuration from YAML

This is the first task in the pipeline. It generates the full config.json
from the simplified config.yaml file, ensuring all subsequent tasks have
the correct configuration.

Purpose:
--------
- Makes config generation part of the automated workflow
- Ensures config.yaml is the single source of truth
- Validates configuration before pipeline starts
- Git-clone friendly (users only edit YAML, never JSON)

Configuration:
--------------
Reads config.yaml from the same directory as this script.

Output:
-------
- config.json (auto-generated, ready for tasks 1-7)
- Validation report

Usage:
------
As Databricks task:
  python task0_generate_config.py --config_yaml_path /Workspace/.../config.yaml

With custom output:
  python task0_generate_config.py --config_yaml_path config.yaml --output_path custom_config.json

As standalone:
  python task0_generate_config.py

Features:
---------
  - Validates YAML syntax
  - Checks required fields
  - Compares with existing config.json (if present)
  - Reports what changed
  - Fails pipeline if critical errors found
"""

import os
import sys
import json
from pathlib import Path

# ================================================================================
# CLI ARGUMENT PARSER
# ================================================================================
def parse_args():
    args = {
        "config_yaml_path": "config.yaml",
        "output_path": "config.json",
        "validate_only": False
    }

    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i].lstrip("--")
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                args[key] = sys.argv[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            i += 1

    return args

# ================================================================================
# IMPORT CONFIG BUILDER
# ================================================================================
# Get the directory of this script
script_dir = Path(__file__).parent.absolute()

# Add to path if config_builder is in same directory
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from config_builder import ConfigBuilder
except ImportError:
    print("❌ ERROR: config_builder.py not found in same directory as this script")
    print(f"   Expected location: {script_dir / 'config_builder.py'}")
    sys.exit(1)

# ================================================================================
# MAIN EXECUTION
# ================================================================================
def main():
    args = parse_args()

    yaml_path = args["config_yaml_path"]
    output_path = args["output_path"]
    validate_only = args.get("validate_only", False)

    print("=" * 80)
    print("TASK 0: Generate Configuration from YAML")
    print("=" * 80)
    print(f"YAML config: {yaml_path}")
    print(f"JSON output: {output_path}")
    print("=" * 80)

    # Check if YAML exists
    if not Path(yaml_path).exists():
        print(f"\n❌ ERROR: YAML config not found: {yaml_path}")
        print("\nExpected config.yaml with structure:")
        print("""
project:
  catalog: your_catalog
  schema: your_schema
  volume_root: /Volumes/...

country:
  iso3: IND

inputs:
  proportions_csv: /path/to/proportions.csv
  tsi_csv: /path/to/tsi.csv
  ...
        """)
        sys.exit(1)

    # Build configuration
    try:
        print("\n📖 Reading YAML configuration...")
        builder = ConfigBuilder(yaml_path)

        print(f"   ✓ Catalog: {builder.catalog}")
        print(f"   ✓ Schema: {builder.schema}")
        print(f"   ✓ Country: {builder.iso3}")

        print("\n🔧 Generating full configuration...")
        full_config = builder.build_full_config()

        # Count generated values
        table_keys = [k for k in full_config.keys() if 'table' in k.lower()]
        path_keys = [k for k in full_config.keys() if 'path' in k.lower() or 'dir' in k.lower() or 'root' in k.lower()]

        print(f"   ✓ Generated {len(full_config)} configuration values")
        print(f"   ✓ Table names: {len(table_keys)}")
        print(f"   ✓ Path values: {len(path_keys)}")

        # Validate required fields
        print("\n✅ Validating configuration...")
        required = ["catalog", "schema", "iso3", "volume_root", "proportions_csv_path", "tsi_csv_path"]
        missing = [k for k in required if not full_config.get(k)]

        if missing:
            print(f"   ❌ Missing required fields: {missing}")
            sys.exit(1)
        else:
            print(f"   ✓ All required fields present")

        # Compare with existing config if present
        if Path(output_path).exists():
            print(f"\n🔍 Comparing with existing {output_path}...")
            try:
                with open(output_path, 'r') as f:
                    old_config = json.load(f)

                # Find differences
                changed = []
                for key in full_config.keys():
                    if key not in old_config:
                        changed.append(f"   + NEW: {key}")
                    elif old_config[key] != full_config[key]:
                        changed.append(f"   ~ CHANGED: {key}")
                        changed.append(f"      OLD: {old_config[key]}")
                        changed.append(f"      NEW: {full_config[key]}")

                for key in old_config.keys():
                    if key not in full_config:
                        changed.append(f"   - REMOVED: {key}")

                if changed:
                    print(f"   ⚠️  Found {len([c for c in changed if c.startswith('   ')])} differences:")
                    for change in changed[:20]:  # Limit output
                        print(change)
                    if len(changed) > 20:
                        print(f"   ... and {len(changed) - 20} more")
                else:
                    print("   ✓ Configuration unchanged")

            except Exception as e:
                print(f"   ⚠️  Could not compare: {e}")

        # Write output (unless validate-only mode)
        if validate_only:
            print("\n🔍 Validation complete (not writing config.json)")
        else:
            print(f"\n💾 Writing configuration to {output_path}...")
            with open(output_path, 'w') as f:
                json.dump(full_config, f, indent=2)

            # Verify written file
            file_size = Path(output_path).stat().st_size
            print(f"   ✓ Written {file_size:,} bytes")

        # Print summary
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        builder.print_summary(full_config)

        print("\n" + "=" * 80)
        print("✅ TASK 0 COMPLETE")
        print("=" * 80)
        print(f"Configuration ready for tasks 1-7")
        print(f"Next task: task1_proportions_to_delta.py --config_path {output_path}")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: Configuration generation failed")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
