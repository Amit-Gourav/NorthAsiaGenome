#!/usr/bin/env python3
"""Portable paleoclimate resistance-change analysis.

Example
-------
python resistance_change.py \
    --climate-dir data/chelsa \
    --ice-dir data/ice6g \
    --output-dir results/resistance_change \
    --download-missing

The code does not use genetic statistics. Demographic dates only select
environmental time slices.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from rasterio.warp import reproject


DEFAULT_TIMES = [21000, 18000, 16000, 12000, 8000, 6000, 3000]
CLIMATE_VARIABLES = ("bio01", "bio12", "bio04")
CHELSA_ROOT = "https://os.zhdk.cloud.switch.ch/chelsav1/chelsa_trace"
ICE6G_DOWNLOAD = (
    "https://sharebox.lsce.ipsl.fr/index.php/s/1RSwVf7afj38cmG/download"
    "?path=%2F&files={filename}"
)

DEFAULT_REGIONS = {
    "WG": {"name": "West Eurasian gateway", "box": [30, 55, 45, 65], "color": "#415A77"},
    "NW": {"name": "NW Siberia", "box": [65, 58, 90, 72], "color": "#7B2CBF"},
    "SW": {"name": "SW Siberia / Altai-Sayan", "box": [80, 48, 100, 56], "color": "#C76D00"},
    "ES": {"name": "East Siberia", "box": [110, 55, 140, 68], "color": "#00897B"},
    "WB": {"name": "West Beringia", "box": [150, 55, 178, 70], "color": "#A23B72"},
    "EA": {"name": "Amur / NE Asia", "box": [120, 40, 140, 52], "color": "#527A32"},
}


@dataclass(frozen=True)
class Grid:
    west: float
    south: float
    east: float
    north: float
    resolution: float

    @property
    def width(self) -> int:
        return int(round((self.east - self.west) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.north - self.south) / self.resolution))

    @property
    def transform(self):
        return from_origin(self.west, self.north, self.resolution, self.resolution)

    @property
    def bounds(self):
        return (self.west, self.south, self.east, self.north)

    @property
    def lons(self):
        return self.west + (np.arange(self.width) + 0.5) * self.resolution

    @property
    def lats(self):
        return self.north - (np.arange(self.height) + 0.5) * self.resolution

    @property
    def coordinates(self):
        return np.meshgrid(self.lons, self.lats)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate ensemble paleoclimate resistance changes without genetic input."
    )
    parser.add_argument("--climate-dir", type=Path, default=Path("data/chelsa"))
    parser.add_argument("--ice-dir", type=Path, default=Path("data/ice6g"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/resistance_change"))
    parser.add_argument(
        "--times",
        default=",".join(str(x) for x in DEFAULT_TIMES),
        help="Comma-separated years BP, ordered from older to younger.",
    )
    parser.add_argument(
        "--bounds",
        default="20,35,180,82",
        help="west,south,east,north in decimal degrees.",
    )
    parser.add_argument("--resolution", type=float, default=0.5, help="Grid resolution in degrees.")
    parser.add_argument("--n-models", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8160)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--regions", type=Path, help="Optional JSON file replacing the default regions.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing CHELSA subsets and ICE-6G_C NetCDF files.",
    )
    return parser.parse_args()


def validate_inputs(args):
    times = [int(x.strip()) for x in args.times.split(",")]
    if len(times) < 2 or any(a <= b for a, b in zip(times[:-1], times[1:])):
        raise ValueError("--times must contain at least two dates ordered from older to younger BP")
    if min(times) < 0 or max(times) > 21000:
        raise ValueError("CHELSA-TraCE21k supports dates from 0 to 21,000 BP")
    if any(bp % 500 != 0 for bp in times):
        raise ValueError("Dates must use the 500-year ICE-6G_C time grid")
    bounds = [float(x.strip()) for x in args.bounds.split(",")]
    if len(bounds) != 4:
        raise ValueError("--bounds must be west,south,east,north")
    grid = Grid(*bounds, resolution=args.resolution)
    if grid.width <= 0 or grid.height <= 0:
        raise ValueError("Invalid bounds or resolution")
    regions = json.loads(args.regions.read_text()) if args.regions else DEFAULT_REGIONS
    return times, grid, regions


def time_label(bp: int) -> str:
    return f"{bp / 1000:g}"


def chelsa_url(variable: str, bp: int) -> str:
    # CHELSA file ID 20 is 0 BP; IDs decrease by one per century backward.
    file_id = 20 - bp // 100
    return f"{CHELSA_ROOT}/bio/CHELSA_TraCE21k_{variable}_{file_id}_V1.0.tif"


def download_file(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {destination.name}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def read_chelsa_remote(url: str, grid: Grid) -> np.ndarray:
    settings = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_TIMEOUT": "120",
    }
    with rasterio.Env(**settings):
        with rasterio.open(url) as src:
            window = from_bounds(*grid.bounds, src.transform).round_offsets().round_lengths()
            return src.read(
                1,
                window=window,
                out_shape=(grid.height, grid.width),
                resampling=Resampling.bilinear,
            ).astype("float32")


def load_climate_layer(
    climate_dir: Path,
    variable: str,
    bp: int,
    grid: Grid,
    download_missing: bool,
) -> np.ndarray:
    path = climate_dir / f"chelsa_{variable}_{bp}bp.npy"
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(
                f"Missing {path}. Re-run with --download-missing or supply the cached NumPy layer."
            )
        climate_dir.mkdir(parents=True, exist_ok=True)
        array = read_chelsa_remote(chelsa_url(variable, bp), grid)
        np.save(path, array)
    array = np.load(path).astype("float32")
    if array.shape != (grid.height, grid.width):
        raise ValueError(
            f"{path} has shape {array.shape}; expected {(grid.height, grid.width)}. "
            "Delete it and recreate it for the requested bounds/resolution."
        )
    return array


def ensure_ice_file(ice_dir: Path, bp: int, download_missing: bool) -> Path:
    filename = f"I6_C.VM5a_10min.{time_label(bp)}.nc"
    path = ice_dir / filename
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(f"Missing {path}; re-run with --download-missing")
        download_file(ICE6G_DOWNLOAD.format(filename=urllib.parse.quote(filename)), path)
    return path


def load_ice_slice(ice_dir: Path, bp: int, grid: Grid, download_missing: bool):
    netcdf = ensure_ice_file(ice_dir, bp, download_missing)
    output = {}
    for variable, method in (
        ("sftlf", Resampling.nearest),
        ("sftgif", Resampling.nearest),
        ("Topo", Resampling.bilinear),
    ):
        source_name = f"netcdf:{netcdf}:{variable}"
        with rasterio.open(source_name) as src:
            destination = np.full((grid.height, grid.width), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs="EPSG:4326",
                src_nodata=1e20,
                dst_transform=grid.transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=method,
            )
            output[variable] = destination
    output["land"] = output["sftlf"] >= 50
    output["ice"] = output["sftgif"] >= 50
    output["passable"] = output["land"] & ~output["ice"] & np.isfinite(output["Topo"])
    return output


def scale01(values: np.ndarray, low: float, high: float, reverse: bool = False):
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    return 1 - scaled if reverse else scaled


def ruggedness(topography: np.ndarray, grid: Grid):
    _, latitude = grid.coordinates
    filled = np.where(np.isfinite(topography), topography, 0.0)
    dy_km = grid.resolution * 111.32
    dx_km = grid.resolution * 111.32 * np.maximum(np.cos(np.deg2rad(latitude)), 0.12)
    gradient_y = np.gradient(filled, axis=0) / dy_km
    gradient_x = np.gradient(filled, axis=1) / dx_km
    return np.hypot(gradient_x, gradient_y).astype("float32")


def build_predictors(climate, ice, times, grid):
    scales = {}
    for variable in CLIMATE_VARIABLES:
        pooled = np.concatenate([climate[(bp, variable)][ice[bp]["passable"]] for bp in times])
        pooled = pooled[np.isfinite(pooled)]
        scales[variable] = tuple(float(x) for x in np.percentile(pooled, [5, 95]))

    terrain = {bp: ruggedness(ice[bp]["Topo"], grid) for bp in times}
    pooled_terrain = np.concatenate([terrain[bp][ice[bp]["passable"]] for bp in times])
    ruggedness_scale = (0.0, float(np.percentile(pooled_terrain, 95)))

    predictors = {}
    for bp in times:
        available = ice[bp]["passable"]
        stack = np.stack(
            [
                scale01(climate[(bp, "bio01")], *scales["bio01"], reverse=True),
                scale01(climate[(bp, "bio12")], *scales["bio12"], reverse=True),
                scale01(climate[(bp, "bio04")], *scales["bio04"]),
                scale01(terrain[bp], *ruggedness_scale),
            ]
        ).astype("float32")
        stack[:, ~available] = np.nan
        predictors[bp] = stack
    return predictors, {"climate_5_95": scales, "ruggedness_0_95": ruggedness_scale}


def write_tiff(path: Path, array: np.ndarray, grid: Grid):
    nodata = -9999.0
    data = np.where(np.isfinite(array), array, nodata).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        predictor=2,
    ) as dst:
        dst.write(data, 1)


def resistance_values(predictor_chunk, weights, gamma):
    weighted = weights @ predictor_chunk
    return 1.0 + 9.0 * np.power(np.clip(weighted, 0, 1), gamma[:, None])


def calculate_change_maps(
    predictors,
    ice,
    times,
    weights,
    gamma,
    grid,
    output_dir,
    chunk_size,
):
    tiff_dir = output_dir / "geotiffs"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    summary_rows = []

    for earlier, later in zip(times[:-1], times[1:]):
        common = ice[earlier]["passable"] & ice[later]["passable"]
        flat_ids = np.where(common.ravel())[0]
        first = predictors[earlier].reshape(4, -1)[:, flat_ids]
        second = predictors[later].reshape(4, -1)[:, flat_ids]
        metrics = {
            name: np.empty(len(flat_ids), dtype="float32")
            for name in (
                "median_change",
                "q10_change",
                "q90_change",
                "probability_decrease",
                "probability_increase",
            )
        }

        for start in range(0, len(flat_ids), chunk_size):
            stop = min(start + chunk_size, len(flat_ids))
            before = resistance_values(first[:, start:stop], weights, gamma)
            after = resistance_values(second[:, start:stop], weights, gamma)
            change = after - before
            metrics["median_change"][start:stop] = np.median(change, axis=0)
            metrics["q10_change"][start:stop] = np.quantile(change, 0.10, axis=0)
            metrics["q90_change"][start:stop] = np.quantile(change, 0.90, axis=0)
            metrics["probability_decrease"][start:stop] = np.mean(change < 0, axis=0)
            metrics["probability_increase"][start:stop] = np.mean(change > 0, axis=0)

        full_maps = {}
        for name, values in metrics.items():
            full = np.full((grid.height, grid.width), np.nan, dtype="float32")
            full.ravel()[flat_ids] = values
            full_maps[name] = full
            write_tiff(
                tiff_dir / f"{earlier}bp_to_{later}bp_{name}.tif",
                full,
                grid,
            )
        results[(earlier, later)] = full_maps
        summary_rows.append(
            {
                "earlier_bp": earlier,
                "later_bp": later,
                "comparable_cells": len(flat_ids),
                "spatial_median_change": float(np.median(metrics["median_change"])),
                "fraction_cells_decrease_p_ge_0_90": float(
                    np.mean(metrics["probability_decrease"] >= 0.90)
                ),
                "fraction_cells_increase_p_ge_0_90": float(
                    np.mean(metrics["probability_increase"] >= 0.90)
                ),
            }
        )
        print(f"completed {earlier / 1000:g} -> {later / 1000:g} ka", flush=True)
    return results, pd.DataFrame(summary_rows)


def region_mask(region, grid):
    longitude, latitude = grid.coordinates
    west, south, east, north = region["box"]
    return (
        (longitude >= west)
        & (longitude <= east)
        & (latitude >= south)
        & (latitude <= north)
    )


def calculate_regional_summaries(
    predictors,
    ice,
    times,
    weights,
    gamma,
    grid,
    regions,
):
    trajectories = {}
    level_rows = []
    change_rows = []

    for code, region in regions.items():
        common = region_mask(region, grid)
        for bp in times:
            common &= ice[bp]["passable"]
        flat_ids = np.where(common.ravel())[0]
        if not len(flat_ids):
            print(f"warning: region {code} has no cells passable at all dates", flush=True)
            continue

        for bp in times:
            data = predictors[bp].reshape(4, -1)[:, flat_ids]
            regional = np.median(resistance_values(data, weights, gamma), axis=1)
            trajectories[(code, bp)] = regional
            level_rows.append(
                {
                    "region": code,
                    "region_name": region["name"],
                    "bp": bp,
                    "common_cells": len(flat_ids),
                    "ensemble_median": float(np.median(regional)),
                    "q10": float(np.quantile(regional, 0.10)),
                    "q90": float(np.quantile(regional, 0.90)),
                }
            )

        for earlier, later in zip(times[:-1], times[1:]):
            change = trajectories[(code, later)] - trajectories[(code, earlier)]
            change_rows.append(
                {
                    "region": code,
                    "region_name": region["name"],
                    "earlier_bp": earlier,
                    "later_bp": later,
                    "median_change": float(np.median(change)),
                    "q10": float(np.quantile(change, 0.10)),
                    "q90": float(np.quantile(change, 0.90)),
                    "probability_decrease": float(np.mean(change < 0)),
                    "probability_increase": float(np.mean(change > 0)),
                }
            )
    return pd.DataFrame(level_rows), pd.DataFrame(change_rows)


def add_regions(ax, regions):
    for code, region in regions.items():
        west, south, east, north = region["box"]
        color = region.get("color", "#333333")
        ax.add_patch(
            Rectangle(
                (west, south),
                east - west,
                north - south,
                fill=False,
                lw=1,
                ec=color,
            )
        )
        ax.text(west + 0.5, north - 0.5, code, color=color, weight="bold", va="top")


def plot_change_maps(results, ice, times, grid, regions, output_dir):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    cmap = LinearSegmentedColormap.from_list(
        "change",
        ["#174A7E", "#6A9BC1", "#D3E3EE", "#F7F7F5", "#F2D0B6", "#D07A4B", "#8C2D1D"],
    )
    all_values = np.concatenate(
        [
            np.abs(maps["median_change"])[np.isfinite(maps["median_change"])]
            for maps in results.values()
        ]
    )
    limit = float(np.clip(np.quantile(all_values, 0.98), 1.0, 3.0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    intervals = list(zip(times[:-1], times[1:]))
    columns = min(3, len(intervals))
    rows = math.ceil(len(intervals) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.25 * columns, 3.1 * rows), squeeze=False)
    axes_flat = list(axes.flat)
    extent = grid.bounds
    longitude, latitude = grid.coordinates

    for ax, interval in zip(axes_flat, intervals):
        earlier, later = interval
        maps = results[interval]
        image = ax.imshow(
            maps["median_change"],
            extent=extent,
            origin="upper",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        decrease = np.ma.masked_invalid(maps["probability_decrease"])
        increase = np.ma.masked_invalid(maps["probability_increase"])
        if np.nanmax(decrease) >= 0.90:
            ax.contour(longitude, latitude, decrease, levels=[0.90], colors="#082F55", linewidths=0.65)
        if np.nanmax(increase) >= 0.90:
            ax.contour(longitude, latitude, increase, levels=[0.90], colors="#7F1D1D", linewidths=0.65)
        # The ICE-6G_C shoreline at the later date provides geographic
        # context while remaining consistent with the paleogeography model.
        ax.contour(
            longitude,
            latitude,
            ice[later]["land"].astype("uint8"),
            levels=[0.5],
            colors="#263238",
            linewidths=0.55,
            alpha=0.90,
        )
        add_regions(ax, regions)
        ax.set_xlim(grid.west, grid.east)
        ax.set_ylim(grid.south, grid.north)
        ax.set_title(f"{earlier / 1000:g} -> {later / 1000:g} ka", loc="left", weight="bold")
        ax.grid(color="white", alpha=0.3, lw=0.4)

    for ax in axes_flat[len(intervals) :]:
        ax.axis("off")
    fig.suptitle("Spatial change in environmental resistance", weight="bold", fontsize=16)
    colorbar = fig.colorbar(image, ax=list(axes.flat), orientation="horizontal", fraction=0.05, pad=0.08)
    colorbar.set_label("Median change in relative resistance (later minus earlier)")
    fig.savefig(output_dir / "resistance_change_maps.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "resistance_change_maps.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    times, grid, regions = validate_inputs(args)
    args.climate_dir.mkdir(parents=True, exist_ok=True)
    args.ice_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    climate = {
        (bp, variable): load_climate_layer(
            args.climate_dir,
            variable,
            bp,
            grid,
            args.download_missing,
        )
        for bp in times
        for variable in CLIMATE_VARIABLES
    }
    ice = {
        bp: load_ice_slice(args.ice_dir, bp, grid, args.download_missing)
        for bp in times
    }
    predictors, scaling = build_predictors(climate, ice, times, grid)

    rng = np.random.default_rng(args.seed)
    weights = rng.dirichlet(np.ones(4), args.n_models).astype("float32")
    gamma = rng.uniform(0.75, 1.50, args.n_models).astype("float32")

    change_maps, map_summary = calculate_change_maps(
        predictors,
        ice,
        times,
        weights,
        gamma,
        grid,
        args.output_dir,
        args.chunk_size,
    )
    levels, changes = calculate_regional_summaries(
        predictors,
        ice,
        times,
        weights,
        gamma,
        grid,
        regions,
    )
    map_summary.to_csv(args.output_dir / "map_change_summary.csv", index=False)
    levels.to_csv(args.output_dir / "regional_resistance_levels.csv", index=False)
    changes.to_csv(args.output_dir / "regional_resistance_changes.csv", index=False)
    plot_change_maps(change_maps, ice, times, grid, regions, args.output_dir)

    metadata = {
        "times_bp": times,
        "bounds": grid.bounds,
        "resolution_degrees": grid.resolution,
        "n_models": args.n_models,
        "seed": args.seed,
        "weights": "Dirichlet(1,1,1,1)",
        "response_exponent": "Uniform(0.75,1.50)",
        "scaling": scaling,
        "regions": regions,
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"outputs written to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
