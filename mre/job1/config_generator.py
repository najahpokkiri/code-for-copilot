"""
Config Generator Helper
Generates minimal configuration YAML for the building enrichment pipeline.
"""

import yaml


def generate_minimal_config(
    iso3: str,
    catalog: str,
    schema: str,
    volume_base: str,
    workspace_base: str,
    proportions_csv: str,
    tsi_csv: str,
    admin_boundaries: str,
    run_mode: str = "full",
    cell_size: int = 2000,
    download_concurrency: int = 3,
    max_workers: int = 8,
    tile_parallelism: int = 4,
    sample_size: int = None,
    max_tiles: int = None
):
    """
    Generate minimal configuration that will be used by Task 0.

    Args:
        iso3: Country ISO3 code
        catalog: Databricks catalog name
        schema: Databricks schema name
        volume_base: Volume root path
        workspace_base: Workspace path where scripts are located
        proportions_csv: Path to proportions CSV file
        tsi_csv: Path to TSI CSV file
        admin_boundaries: Path to admin boundaries file
        run_mode: "test" or "full" (default: "full")
        cell_size: Grid cell size in meters (default: 2000)
        download_concurrency: Parallel tile downloads (default: 3)
        max_workers: Raster processing threads (default: 8)
        tile_parallelism: Concurrent tile processing (default: 4)
        sample_size: Optional sample size for test mode
        max_tiles: Optional max tiles for test mode

    Returns:
        dict: Minimal configuration dictionary
    """
    config = {
        "project": {
            "catalog": catalog,
            "schema": schema,
            "volume_root": volume_base
        },
        "country": {
            "iso3": iso3
        },
        "inputs": {
            "proportions_csv": proportions_csv,
            "tsi_csv": tsi_csv,
            "admin_boundaries": admin_boundaries,
            "tile_footprint": f"{workspace_base}/data/ghsl2_0_mwd_l1_tile_schema_land.gpkg"
        },
        "params": {
            "cell_size": cell_size,
            "export_crs": "EPSG:4326",
            "target_crs": "ESRI:54009",
            "datasets": "built_c,smod",
            "download_concurrency": download_concurrency,
            "download_retries": 2,
            "use_smod": True,
            "use_boundary_mask": True,
            "include_nodata": True,
            "add_percentages": False,
            "chunk_size": 10000,
            "max_workers": max_workers,
            "tile_parallelism": tile_parallelism,
            "sample_size": sample_size if sample_size else 10000,
            "max_tiles": max_tiles if max_tiles else None,
            "stage_to_local": True,
            "local_dir": "/local_disk0/raster_cache",
            "spark_tmp_dir": "/tmp/job3_grid_tmp"
        },
        "flags": {
            "dry_run": False,
            "preview": True,
            "preview_rows": 5,
            "overwrite_schema": True,
            "write_mode": "overwrite",
            "csv_infer_schema": True,
            "save_temp_csv": False,
            "save_per_tile": False
        },
        "workspace_base": workspace_base,
        "run_mode": run_mode.lower()
    }

    return config


def save_config_to_workspace(config: dict, iso3: str, workspace_base: str, dbutils):
    """
    Save minimal config to workspace as YAML file.

    Args:
        config: Configuration dictionary
        iso3: Country ISO3 code
        workspace_base: Workspace path
        dbutils: Databricks utilities object

    Returns:
        str: Path to saved config in workspace
    """
    # Save to local temp first
    temp_config_path = f"/tmp/minimal_config_{iso3}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    # Upload to workspace
    workspace_config_path = f"{workspace_base}/temp_minimal_config_{iso3}.yaml"
    dbutils.fs.cp(f"file:{temp_config_path}", f"file:{workspace_config_path}", recurse=True)

    return workspace_config_path
