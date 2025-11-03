#!/usr/bin/env python3
"""Visual documentation for the Databricks geospatial job (6 sequential tasks).

This script renders an India-themed process visualization that can be embedded in
runbooks, READMEs, or slide decks. The output figure layers the six tasks on top
of an India map silhouette, groups related work into four thematic quadrants, and
includes an inset chart that depicts the sequential execution flow.

Usage
-----
    python india_pipeline_visual.py

Dependencies
------------
    pip install geopandas matplotlib

The script relies on the Natural Earth 1:110m dataset that ships with GeoPandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import patches


OUTPUT_PATH = Path(__file__).with_name("india_geospatial_pipeline.png")


# Task metadata: title, subtitle, relative position (0-1 scale), marker color.
TASKS: List[Dict[str, object]] = [
    {
        "name": "Task 1",
        "title": "Proportions ➜ Delta",
        "details": "Normalize floor-count CSVs into Delta tables",
        "rel_pos": (0.30, 0.80),
        "color": "#FF9933",  # Saffron
    },
    {
        "name": "Task 2",
        "title": "Grid Generation",
        "details": "Clip 5 km grid to ISO3 and persist centroids",
        "rel_pos": (0.62, 0.80),
        "color": "#FFFFFF",  # White
    },
    {
        "name": "Task 3",
        "title": "Tile Downloader",
        "details": "Fetch GHSL built_c & SMOD tiles; log status",
        "rel_pos": (0.25, 0.56),
        "color": "#138808",  # India green
    },
    {
        "name": "Task 4",
        "title": "Raster Stats",
        "details": "Compute class counts per grid window",
        "rel_pos": (0.62, 0.53),
        "color": "#0B3D2E",  # Deep green accent
    },
    {
        "name": "Task 5",
        "title": "Post-Processing",
        "details": "Aggregate sector totals + QA snapshots",
        "rel_pos": (0.40, 0.28),
        "color": "#0E5E6F",
    },
    {
        "name": "Task 6",
        "title": "Create Views",
        "details": "Publish RES/COM/IND TSI proportion views",
        "rel_pos": (0.66, 0.26),
        "color": "#1C658C",
    },
]


# Quadrant overlays (title, description, RGBA color, relative anchor).
QUADRANTS: List[Dict[str, object]] = [
    {
        "title": "Data Foundations",
        "summary": "Reference proportions & governance",
        "color": (1.0, 0.60, 0.2, 0.18),
        "rel_anchor": (0.00, 0.50),  # bottom-left corner of top-left quadrant
        "label_offset": (0.22, 0.94),
    },
    {
        "title": "Spatial Preparation",
        "summary": "Grid creation & geospatial triggers",
        "color": (1.0, 1.0, 1.0, 0.18),
        "rel_anchor": (0.50, 0.50),  # top-right
        "label_offset": (0.77, 0.94),
    },
    {
        "title": "Raster Analytics",
        "summary": "Tile acquisition & per-cell metrics",
        "color": (0.19, 0.53, 0.31, 0.20),
        "rel_anchor": (0.00, 0.00),  # bottom-left
        "label_offset": (0.22, 0.32),
    },
    {
        "title": "Insights & Sharing",
        "summary": "Sector aggregation & curated views",
        "color": (0.09, 0.35, 0.49, 0.20),
        "rel_anchor": (0.50, 0.00),
        "label_offset": (0.77, 0.32),
    },
]


def _relative_to_data(rel_x: float, rel_y: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Convert values in 0..1 space into map coordinates using the India bounds."""

    minx, miny, maxx, maxy = bounds
    span_x = maxx - minx
    span_y = maxy - miny
    return minx + rel_x * span_x, miny + rel_y * span_y


def _plot_india(ax: plt.Axes) -> Tuple[float, float, float, float]:
    """Plot the India outline and return its bounding box."""

    india = _load_india_geometry()
    india.plot(ax=ax, color="#f4efe6", edgecolor="#3d2f1d", linewidth=1.1, zorder=1)
    bounds = india.total_bounds  # (minx, miny, maxx, maxy)
    margin_x = (bounds[2] - bounds[0]) * 0.05
    margin_y = (bounds[3] - bounds[1]) * 0.05
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)
    ax.set_aspect("equal")
    ax.axis("off")
    return bounds


def _load_india_geometry() -> gpd.GeoDataFrame:
    """Load the India polygon from GeoPandas' naturalearth_lowres dataset."""

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    india = world[world["name"] == "India"].to_crs(epsg=4326)
    if india.empty:
        raise RuntimeError("Could not locate India geometry in naturalearth_lowres dataset.")
    return india


def _add_quadrants(ax: plt.Axes, bounds: Tuple[float, float, float, float]) -> None:
    """Overlay translucent quadrants with thematic labels."""

    minx, miny, maxx, maxy = bounds
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2
    width = maxx - minx
    height = maxy - miny

    quadrants_xy = {
        (0.00, 0.50): (minx, midy),  # top-left
        (0.50, 0.50): (midx, midy),  # top-right
        (0.00, 0.00): (minx, miny),  # bottom-left
        (0.50, 0.00): (midx, miny),  # bottom-right
    }

    for quad in QUADRANTS:
        anchor = quadrants_xy[quad["rel_anchor"]]  # type: ignore[index]
        rect = patches.Rectangle(
            anchor,
            width * 0.5,
            height * 0.5,
            linewidth=0,
            facecolor=quad["color"],  # type: ignore[arg-type]
            zorder=2,
        )
        ax.add_patch(rect)
        label_point = _relative_to_data(quad["label_offset"][0], quad["label_offset"][1], bounds)  # type: ignore[index]
        ax.text(
            label_point[0],
            label_point[1],
            f"{quad['title']}\n{quad['summary']}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#2d1e0f",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.65, edgecolor="none"),
            zorder=3,
        )


def _add_tasks(ax: plt.Axes, bounds: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    """Plot task markers and annotations. Returns data coordinates for each task."""

    coords = []
    span_y = bounds[3] - bounds[1]
    text_offset = span_y * 0.035

    for task in TASKS:
        x, y = _relative_to_data(task["rel_pos"][0], task["rel_pos"][1], bounds)  # type: ignore[index]
        coords.append((x, y))

        # Marker representing the task
        circle = patches.Circle(
            (x, y),
            radius=span_y * 0.018,
            facecolor=task["color"],
            edgecolor="#1b1b1b",
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(circle)

        ax.text(
            x,
            y + text_offset,
            task["name"],
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1b1b1b",
            zorder=6,
        )

        ax.text(
            x,
            y - text_offset,
            f"{task['title']}\n{task['details']}",
            ha="center",
            va="top",
            fontsize=9.5,
            color="#1b1b1b",
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    return coords


def _link_tasks(ax: plt.Axes, coords: List[Tuple[float, float]]) -> None:
    """Draw arrows to emphasize sequential flow."""

    for idx in range(len(coords) - 1):
        start = coords[idx]
        end = coords[idx + 1]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#07492f", linewidth=2.2, shrinkA=8, shrinkB=8),
            zorder=4,
        )


def _add_inset_chart(fig: plt.Figure) -> None:
    """Create an inset chart that reads like a run summary graph."""

    ax_inset = fig.add_axes([0.60, 0.04, 0.32, 0.23])
    task_indices = list(range(1, len(TASKS) + 1))
    completion = [5, 20, 45, 72, 88, 100]

    ax_inset.plot(
        task_indices,
        completion,
        color="#138808",
        linewidth=2.2,
        marker="o",
        markersize=6,
        markerfacecolor="#FF9933",
        markeredgecolor="#0b3d2e",
    )
    ax_inset.fill_between(task_indices, completion, color="#138808", alpha=0.08)

    ax_inset.set_xticks(task_indices)
    ax_inset.set_xticklabels([task["name"] for task in TASKS], rotation=30, ha="right", fontsize=8.5)
    ax_inset.set_ylabel("Cumulative Progress (%)", fontsize=9)
    ax_inset.set_ylim(0, 105)
    ax_inset.set_xlim(0.8, len(TASKS) + 0.2)
    ax_inset.set_title("Process Run-through", fontsize=11, fontweight="bold")
    ax_inset.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_inset.set_facecolor("#fbfaf7")

    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    ax_inset.text(
        0.02,
        0.88,
        "Key Outputs\n• Delta tables per task\n• CSV + View snapshots",
        transform=ax_inset.transAxes,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#d0c7b8"),
    )


def _add_title(fig: plt.Figure) -> None:
    """Add a headline and subtitle to the visualization."""

    fig.text(
        0.08,
        0.95,
        "India Geospatial Solutions Pipeline",
        fontsize=18,
        fontweight="bold",
        color="#1b1b1b",
    )
    fig.text(
        0.08,
        0.92,
        "Six-task Databricks job (sequential) with India-first storytelling",
        fontsize=12,
        color="#44403c",
    )
    fig.text(
        0.08,
        0.07,
        "Artifacts: proportions, grid centroids, download statuses, raster counts, sector estimates, TSI views",
        fontsize=9.5,
        color="#40362d",
    )


def build_pipeline_visual(output_path: Path = OUTPUT_PATH) -> Path:
    """Generate the India-themed visualization and write it to ``output_path``."""

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor("#f8f5f0")
    fig.patch.set_facecolor("white")

    bounds = _plot_india(ax)
    _add_quadrants(ax, bounds)
    coords = _add_tasks(ax, bounds)
    _link_tasks(ax, coords)
    _add_inset_chart(fig)
    _add_title(fig)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    build_pipeline_visual()

