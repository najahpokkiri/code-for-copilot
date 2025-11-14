#!/usr/bin/env python3
"""
Utility helpers to ensure Python dependencies are available when tasks run on an
existing Databricks cluster (where job-level libraries are not auto-installed).

Each task imports and calls ``ensure_runtime_dependencies()`` before importing
third-party libraries. The helper installs packages from requirements.txt once
per cluster (tracked via a hash stored in /tmp) and skips redundant installs on
subsequent tasks.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict

# Maps import names to the package spec that satisfies them.
DEFAULT_PACKAGES: Dict[str, str] = {
    "geopandas": "geopandas==0.14.0",
    "shapely": "shapely==2.0.2",
    "rasterio": "rasterio==1.3.9",
    "pandas": "pandas==2.0.3",
    "numpy": "numpy==1.24.3",
    "pyarrow": "pyarrow==13.0.0",
    "requests": "requests==2.31.0",
    "yaml": "pyyaml==6.0.1",
}

_SENTINEL_PATH = Path("/tmp/mre_job1_requirements.sha256")


def _hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def ensure_runtime_dependencies(packages: Dict[str, str] | None = None) -> None:
    """
    Install required packages if they are missing or if requirements.txt changed.

    Args:
        packages: Optional override mapping import name -> package spec. Defaults
                  to the shared pipeline requirements.
    """
    required = packages or DEFAULT_PACKAGES
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    expected_hash = _hash_file(requirements_path) if requirements_path.exists() else ""
    sentinel_hash = _SENTINEL_PATH.read_text().strip() if _SENTINEL_PATH.exists() else ""

    missing = []
    for module_name, package_spec in required.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_spec)

    needs_install = bool(missing)
    if expected_hash and sentinel_hash != expected_hash:
        needs_install = True

    if not needs_install:
        return

    if requirements_path.exists():
        print("📦 Installing pipeline requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        if expected_hash:
            _SENTINEL_PATH.write_text(expected_hash)
    elif missing:
        print(f"📦 Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        if expected_hash:
            _SENTINEL_PATH.write_text(expected_hash)
