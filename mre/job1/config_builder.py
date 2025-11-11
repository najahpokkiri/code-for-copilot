#!/usr/bin/env python3
"""
Configuration Builder for Geospatial Solutions Pipeline

Generates a complete config.json from a simplified config.yaml file.
Auto-creates all derived paths, table names, and folder structures.

Usage:
    python config_builder.py config.yaml
    python config_builder.py config.yaml --output config.json
    python config_builder.py config.yaml --validate  # Compare with existing config

Features:
    - Automatic path derivation from root directories
    - Consistent table naming with ISO3 suffix
    - Folder structure validation
    - Side-by-side comparison with existing config
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from collections import OrderedDict


class ConfigBuilder:
    """Builds complete configuration from simplified YAML"""

    def __init__(self, yaml_path: str):
        """
        Initialize builder with YAML configuration

        Args:
            yaml_path: Path to simplified config.yaml file
        """
        with open(yaml_path, 'r') as f:
            self.simple_config = yaml.safe_load(f)

        # Extract core values
        self.catalog = self.simple_config['project']['catalog']
        self.schema = self.simple_config['project']['schema']
        self.volume_root = self.simple_config['project']['volume_root']
        self.iso3 = self.simple_config['country']['iso3']

        # Extract sections
        self.inputs = self.simple_config['inputs']
        self.params = self.simple_config['params']
        self.flags = self.simple_config['flags']

    def _table_name(self, base_name: str) -> str:
        """
        Build fully-qualified table name with ISO3 suffix

        Args:
            base_name: Base table name (e.g., 'grid_centroids')

        Returns:
            Fully-qualified table name: catalog.schema.basename_ISO3
        """
        return f"{self.catalog}.{self.schema}.{base_name}_{self.iso3}"

    def _volume_path(self, *parts: str) -> str:
        """
        Build volume path from components

        Args:
            *parts: Path components to join

        Returns:
            Absolute path string
        """
        return str(Path(self.volume_root, *parts))

    def build_full_config(self) -> Dict[str, Any]:
        """
        Build complete configuration with all derived values

        Returns:
            Complete configuration dictionary matching original config.json structure
        """
        config = {
            # ========== CORE SETTINGS ==========
            "catalog": self.catalog,
            "schema": self.schema,
            # Execution mode (enables test-mode limits in tasks)
            "run_mode": self.simple_config.get("run_mode", "full"),

            # ========== INPUT FILES ==========
            "proportions_csv_path": self.inputs['proportions_csv'],
            "tsi_csv_path": self.inputs['tsi_csv'],
            "csv_infer_schema": self.flags['csv_infer_schema'],

            # ========== TASK 1: DELTA TABLES ==========
            "proportions_path": self._table_name("building_enrichment_proportions_input"),
            "proportions_table": self._table_name("building_enrichment_proportions_input"),
            "tsi_table": self._table_name("building_enrichment_tsi_input"),

            # ========== COUNTRY SETTINGS ==========
            "iso3": self.iso3,

            # ========== TASK 2: GRID GENERATION ==========
            "grid_output_csv": self._volume_path("output", "grid_centroids.csv"),
            "delta_table_base": self._table_name("grid_centroids"),
            "cell_size": self.params['cell_size'],
            "export_crs": self.params['export_crs'],
            "dry_run": self.flags['dry_run'],

            # ========== TASK 3: TILE DOWNLOADER ==========
            "tiles_dest_root": self._volume_path("inputs", "tiles"),
            "download_status_table": self._table_name("download_status"),
            "datasets": self.params['datasets'],
            "download_concurrency": self.params['download_concurrency'],
            "download_retries": self.params['download_retries'],
            "spark_tmp_dir": self.params['spark_tmp_dir'],
            # Concurrency and sampling controls
            "tile_parallelism": self.params['tile_parallelism'],
            # Keep both for compatibility; tasks currently ignore this
            "sample_size": self.params.get('sample_size'),
            "SAMPLE_SIZE": self.params.get('sample_size'),
            # Optional: cap number of tiles in test mode
            "max_tiles": self.params.get('max_tiles'),

            # ========== TASK 4: RASTER STATS ==========
            "overwrite_schema": self.flags['overwrite_schema'],
            "preview": self.flags['preview'],
            "preview_rows": self.flags['preview_rows'],
            "grid_source": self._table_name("grid_centroids"),
            "built_root": self._volume_path("inputs", "tiles", "built_c"),
            "smod_root": self._volume_path("inputs", "tiles", "smod"),
            "output_dir": self._volume_path("outputs"),
            "counts_delta_table": self._table_name("grid_counts"),
            "use_smod": self.params['use_smod'],
            "include_nodata": self.params['include_nodata'],
            "add_percentages": self.params['add_percentages'],
            "use_boundary_mask": self.params['use_boundary_mask'],

            # Admin boundaries
            "admin_path": self.inputs['admin_boundaries'],
            "admin_field": "ISO3",
            "admin_value": self.iso3,

            # Tile footprints
            "tile_footprint_path": self.inputs['tile_footprint'],
            "tile_id_field": "tile_id",

            # Coordinate systems
            "target_crs": self.params['target_crs'],

            # Performance
            "chunk_size": self.params['chunk_size'],
            "max_workers": self.params['max_workers'],
            "stage_to_local": self.params['stage_to_local'],
            "local_dir": self.params['local_dir'],
            "save_per_tile": self.flags['save_per_tile'],
            "write_mode": self.flags['write_mode'],

            # ========== TASK 5: POST-PROCESSING ==========
            "grid_count_table": self._table_name("grid_counts"),
            "output_table": self._table_name("building_enrichment_output"),
            "save_temp_csv": self.flags['save_temp_csv'],
        }

        return config

    def generate_folder_list(self) -> list:
        """
        Generate list of folders that will be created/used

        Returns:
            List of folder paths
        """
        return [
            self._volume_path("inputs", "tiles", "built_c"),
            self._volume_path("inputs", "tiles", "smod"),
            self._volume_path("inputs", "admin"),
            self._volume_path("inputs", "multipliers"),
            self._volume_path("outputs"),
        ]

    def print_summary(self, config: Dict[str, Any]):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        print(f"\n📊 Project:")
        print(f"   Catalog: {config['catalog']}")
        print(f"   Schema: {config['schema']}")
        print(f"   Country: {config['iso3']}")

        print(f"\n📁 Volume Root:")
        print(f"   {self.volume_root}")

        print(f"\n📋 Delta Tables Generated:")
        tables = [k for k, v in config.items() if 'table' in k and isinstance(v, str) and '.' in v]
        for table_key in tables:
            print(f"   • {table_key}: {config[table_key]}")

        print(f"\n🗂️  Folder Structure:")
        for folder in self.generate_folder_list():
            print(f"   • {folder}")

        print(f"\n⚙️  Key Parameters:")
        print(f"   • Cell Size: {config['cell_size']}m")
        print(f"   • Datasets: {config['datasets']}")
        print(f"   • Use SMOD: {config['use_smod']}")
        print(f"   • Chunk Size: {config['chunk_size']}")
        print(f"   • Max Workers: {config['max_workers']}")
        print("="*80 + "\n")


def compare_configs(new_config: Dict[str, Any], old_path: str):
    """
    Compare generated config with existing config

    Args:
        new_config: Newly generated configuration
        old_path: Path to existing config.json
    """
    try:
        with open(old_path, 'r') as f:
            old_config = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  No existing config found at {old_path}")
        return

    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)

    # Find differences
    all_keys = set(old_config.keys()) | set(new_config.keys())

    missing_in_new = set(old_config.keys()) - set(new_config.keys())
    missing_in_old = set(new_config.keys()) - set(old_config.keys())
    different_values = []

    for key in all_keys:
        if key in old_config and key in new_config:
            if old_config[key] != new_config[key]:
                different_values.append((key, old_config[key], new_config[key]))

    if not missing_in_new and not missing_in_old and not different_values:
        print("\n✅ Configurations are identical!")
    else:
        if missing_in_new:
            print("\n❌ Keys in old config but missing in new:")
            for key in sorted(missing_in_new):
                print(f"   • {key}: {old_config[key]}")

        if missing_in_old:
            print("\n➕ New keys not in old config:")
            for key in sorted(missing_in_old):
                print(f"   • {key}: {new_config[key]}")

        if different_values:
            print("\n⚠️  Keys with different values:")
            for key, old_val, new_val in different_values:
                print(f"   • {key}:")
                print(f"      OLD: {old_val}")
                print(f"      NEW: {new_val}")

    print("="*80 + "\n")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python config_builder.py <config.yaml> [--output config.json] [--validate]")
        print("\nOptions:")
        print("  --output FILE    Write output to FILE (default: config.json)")
        print("  --validate       Compare with existing config.json")
        print("  --summary        Show configuration summary only (no file output)")
        sys.exit(1)

    yaml_path = sys.argv[1]

    if not Path(yaml_path).exists():
        print(f"❌ Error: {yaml_path} not found!")
        sys.exit(1)

    # Parse arguments
    output_path = "config.json"
    validate = False
    summary_only = False

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--validate":
            validate = True
            i += 1
        elif sys.argv[i] == "--summary":
            summary_only = True
            i += 1
        else:
            i += 1

    # Build configuration
    try:
        print(f"📖 Reading configuration from: {yaml_path}")
        builder = ConfigBuilder(yaml_path)
        full_config = builder.build_full_config()

        # Show summary
        builder.print_summary(full_config)

        # Write output
        if not summary_only:
            with open(output_path, 'w') as f:
                json.dump(full_config, f, indent=2)
            print(f"✅ Generated configuration written to: {output_path}")

        # Validate if requested
        if validate:
            old_config_path = str(Path(yaml_path).parent / "config.json")
            compare_configs(full_config, old_config_path)

    except Exception as e:
        print(f"\n❌ Error building configuration:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
