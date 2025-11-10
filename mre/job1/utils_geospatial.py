"""
Shared geospatial IO helpers for the MRE job pipeline.

Provides utilities to:
  * Resolve DBFS-style URIs to filesystem paths accessible to local libraries
  * Maintain a persistent decompressed copy of gzipped GeoJSON-like vectors
  * Read vector files with sensible GDAL environment defaults
"""

from __future__ import annotations

import gzip
import os
import shutil
from typing import Optional, Dict, Any

import fiona
import geopandas as gpd


def normalize_path(path: Optional[str]) -> Optional[str]:
    """
    Convert Databricks DBFS URIs (dbfs:/...) to their mounted filesystem path (/dbfs/...).
    Returns the input unchanged for any other scheme.
    """
    if path is None:
        return None
    if path.startswith("dbfs:"):
        return path.replace("dbfs:", "/dbfs", 1)
    return path


def ensure_uncompressed_geojson(path: str, cache_dir: Optional[str] = None, verbose: bool = False) -> str:
    """
    If `path` points to a gzip-compressed GeoJSON (or other json-like vector), decompress it
    to a sidecar file (re-using a cached copy when possible) and return the decompressed path.

    Parameters
    ----------
    path : str
        The source path, possibly starting with dbfs:/ and possibly ending with .gz.
    cache_dir : Optional[str]
        Optional directory to place decompressed files. Defaults to the same directory as the
        source file. The directory will be created if it does not exist.
    verbose : bool
        If True, prints out status messages while decompressing/reusing cache.
    """
    fs_path = normalize_path(path)
    if fs_path is None:
        raise RuntimeError("Path must be provided for decompression.")

    if not fs_path.lower().endswith(".gz"):
        return fs_path

    env_cache = os.environ.get("GEOJSON_CACHE_DIR")
    default_dir = os.path.dirname(fs_path) or "/tmp"
    dest_dir = cache_dir or env_cache or default_dir

    def _ensure_dir(path: str) -> str:
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError:
            if cache_dir:
                raise
            fallback = env_cache or "/dbfs/tmp/geospatial_cache"
            if path == fallback:
                raise
            os.makedirs(fallback, exist_ok=True)
            return fallback

    dest_dir = _ensure_dir(dest_dir)
    base_name = os.path.basename(fs_path)[:-3]  # strip .gz
    target_path = os.path.join(dest_dir, base_name)

    def _needs_refresh(src: str, dst: str) -> bool:
        if not os.path.exists(dst):
            return True
        try:
            return os.path.getmtime(dst) < os.path.getmtime(src)
        except OSError:
            return True

    if _needs_refresh(fs_path, target_path):
        if verbose:
            print(f"Decompressing {fs_path} -> {target_path}")
        with gzip.open(fs_path, "rb") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        if verbose:
            print(f"Reusing cached decompressed file at {target_path}")

    return target_path


def read_vector_file(path: str, *, cache_dir: Optional[str] = None, verbose: bool = False,
                     driver: Optional[str] = None, fiona_env_opts: Optional[Dict[str, Any]] = None) -> gpd.GeoDataFrame:
    """
    Read a vector dataset from the given path, handling DBFS URIs and gzipped GeoJSON automatically.

    Parameters
    ----------
    path : str
        Source path to read. Supports dbfs:/ URIs and gzipped GeoJSON.
    cache_dir : Optional[str]
        Optional directory used when caching decompressed copies.
    verbose : bool
        If True, prints status messages from the helper functions.
    driver : Optional[str]
        Explicit Fiona driver to use. When omitted, GDAL will infer from the filename.
    fiona_env_opts : Optional[Dict[str, Any]]
        Extra kwargs forwarded to `fiona.Env`.
    """
    env_opts: Dict[str, Any] = {"OGR_GEOJSON_MAX_OBJ_SIZE": 0}
    if fiona_env_opts:
        env_opts.update(fiona_env_opts)

    local_path = ensure_uncompressed_geojson(path, cache_dir=cache_dir, verbose=verbose)

    with fiona.Env(**env_opts):
        return gpd.read_file(local_path, driver=driver)
