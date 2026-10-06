"""Explicit model adapters and QC-processed glider readers."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .core import depth_edges, density
from .forecast import prepare_forecast

LOGGER = logging.getLogger(__name__)


def expand(patterns: str | list[str], root: Path) -> list[Path]:
    """Resolve configured paths/globs, failing on an empty input."""
    patterns = [patterns] if isinstance(patterns, str) else patterns
    paths: set[Path] = set()
    for pattern in patterns:
        p = Path(pattern).expanduser()
        paths.update(
            Path(x).resolve()
            for x in glob.glob(str(p if p.is_absolute() else root / p))
        )
    if not paths:
        raise FileNotFoundError(f"No inputs matched {patterns}")
    return sorted(paths)


def open_model(
    paths: list[Path], spec: dict[str, Any], max_depth: float
) -> xr.Dataset:
    """Normalize supplied z-level archives, checking separable geographic grids.

    Native ROMS sigma coordinates require a preceding physical-z conversion;
    the supplied NCSU/CICESE archives are already on the HYCOM z grid.
    """
    names = spec["variables"]
    matlab = spec.get("time_encoding") == "matlab_datenum"
    options = {
        "decode_times": not matlab,
        "engine": "netcdf4",
        "chunks": {
            spec.get("forecast", {}).get("lead_coordinate", names["time"]): 1
        },
    }
    if len(paths) == 1:
        source = xr.open_dataset(paths[0], **options)
    else:
        source = xr.open_mfdataset(
            paths,
            combine="nested",
            concat_dim=spec.get("forecast", {}).get(
                "concat_dim", names["time"]
            ),
            parallel=False,
            **options,
        )
    source = prepare_forecast(source, spec)
    lon = np.asarray(source[names["lon"]].values, dtype=float)
    lat = np.asarray(source[names["lat"]].values, dtype=float)
    if lon.ndim == lat.ndim == 2:
        if not (
            np.allclose(lon, lon[0:1, :], atol=1e-6, equal_nan=True)
            and np.allclose(lat, lat[:, 0:1], atol=1e-6, equal_nan=True)
        ):
            source.close()
            raise ValueError(
                "Nonseparable curvilinear grid: supply a physical lat/lon z-level product"
            )
        yd, xd = source[names["lon"]].dims
        lon, lat = lon[0], lat[:, 0]
    elif lon.ndim == lat.ndim == 1:
        yd, xd = source[names["lat"]].dims[0], source[names["lon"]].dims[0]
    else:
        raise ValueError(
            "Unsupported longitude/latitude coordinate dimensions"
        )
    depth = np.asarray(source[names["depth"]].values, dtype=float)
    if depth.ndim != 1:
        raise ValueError(
            "Model physical depth must be one dimensional (metres)"
        )
    times = source[names["time"]].values
    if matlab:
        times = pd.to_datetime(
            np.asarray(times, float) - 719529.0, unit="D", origin="unix"
        ).values
        times = times.astype("datetime64[s]").astype("datetime64[ns]")
    if not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError(
            "Undecoded model time; configure the correct time encoding"
        )
    dims = [
        source[names["time"]].dims[0],
        source[names["depth"]].dims[0],
        yd,
        xd,
    ]
    data = {
        v: (
            ("time", "depth", "lat", "lon"),
            source[names[v]].transpose(*dims).data,
        )
        for v in ("temp", "salt")
    }
    ds = xr.Dataset(
        data,
        coords={
            "time": times,
            "depth": np.abs(depth),
            "lat": lat,
            "lon": (lon + 180) % 360 - 180,
        },
    )
    ds.set_close(source.close)
    ds = ds.sortby(["time", "depth", "lat", "lon"])
    for coord in ("time", "depth", "lat", "lon"):
        if np.unique(ds[coord]).size != ds.sizes[coord]:
            raise ValueError(
                f"Duplicate model {coord} values; resolve source overlaps explicitly"
            )
    ds = ds.sel(depth=slice(1, max_depth))
    if "mask_rho" in source:
        mask = xr.DataArray(
            source.mask_rho.values,
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": (lon + 180) % 360 - 180},
        )
        ds = ds.where(mask.sortby(["lat", "lon"]) > 0)
    for var in ("temp", "salt"):
        ds[var] = ds[var].where(abs(ds[var]) < 1000)
    ds.attrs.update(
        model=spec["id"], temperature_kind=spec["temperature_kind"]
    )
    if spec.get("forecast"):
        ds.attrs.update(
            {
                key: value
                for key, value in source.attrs.items()
                if key.startswith("forecast_")
            }
        )
    ds.set_close(source.close)
    return ds


def observation_paths(
    spec: dict[str, Any], root: Path, stats_policy: str
) -> tuple[list[Path], str]:
    """Prefer exact rebinning from a requested 6hrstats product's raw companion."""
    requested = expand(spec["paths"], root)
    kind = spec.get("kind", "auto")
    if kind == "auto":
        with xr.open_dataset(requested[0]) as ds:
            kind = (
                "original"
                if ds["depth"].dims == ds["time"].dims
                else "6hrstats"
            )
    if kind == "6hrstats" and stats_policy == "raw_companion":
        originals = [
            p.with_name(
                p.name.replace("_6hrstats.nc", "_original_resolution.nc")
            )
            for p in requested
        ]
        if any(not p.exists() or p == q for p, q in zip(originals, requested)):
            raise ValueError(
                "Exact rebinning needs *_original_resolution.nc. Missing counts/bounds in "
                "6hrstats prevent exact rebinning. Use explicit stats_policy=center_mean "
                "only if averaging source-bin means is intended."
            )
        LOGGER.info(
            "6hrstats requested: using original-resolution companions for exact centre bins"
        )
        return originals, "original"
    return requested, kind


def read_original(
    paths: list[Path], spec: dict[str, Any], settings: dict[str, Any]
) -> xr.Dataset:
    """Read processed QC products, retaining missing values and available QC flags."""
    pieces: list[xr.Dataset] = []
    names = spec.get("variables", {})
    defaults = {
        "temp": "temperature",
        "salt": "salinity",
        "lon": "lon",
        "lat": "lat",
        "depth": "depth",
        "time": "time",
    }
    for path in paths:
        with xr.open_dataset(path) as source:
            values = {
                key: source[names.get(key, default)].values.ravel()
                for key, default in defaults.items()
            }
            for key, stem in (("temp", "temperature"), ("salt", "salinity")):
                for flag in (stem + "_qc", stem + "_qartod_summary_flag"):
                    if (
                        flag in source
                        and source[flag].size == values[key].size
                    ):
                        values[key] = np.where(
                            source[flag].values.ravel() == 1,
                            values[key],
                            np.nan,
                        )
                        break
        pieces.append(
            xr.Dataset({k: ("sample", v) for k, v in values.items()})
        )
    ds = xr.concat(pieces, dim="sample").sortby("time")
    ok = (
        ~np.isnat(ds.time.values)
        & np.isfinite(ds.depth.values)
        & np.isfinite(ds.lat.values)
        & np.isfinite(ds.lon.values)
        & (np.abs(ds.lat.values) <= 90)
        & (np.abs(ds.lon.values) <= 360)
        & (ds.depth.values >= settings["min_depth_m"])
        & (ds.depth.values <= settings["max_depth_m"])
    )
    for key, comparison in (
        ("start", np.greater_equal),
        ("end", np.less_equal),
    ):
        if spec.get(key):
            ok &= comparison(ds.time.values, np.datetime64(spec[key]))
    exclusions: list[dict[str, Any]] = []
    for exclusion in spec.get("exclude_intervals", []):
        start = np.datetime64(exclusion["start_utc"])
        end = np.datetime64(exclusion["end_utc"])
        if np.isnat(start) or np.isnat(end) or start > end:
            raise ValueError(
                "Invalid inclusive observation exclusion interval"
            )
        rejected = (ds.time.values >= start) & (ds.time.values <= end)
        count = int(rejected.sum())
        expected = exclusion.get("expected_samples")
        if expected is not None and count != expected:
            raise ValueError(
                f"Exclusion expected {expected} samples, found {count}; "
                "review the input revision before proceeding"
            )
        ok &= ~rejected
        exclusions.append({**exclusion, "excluded_samples": count})
        LOGGER.warning(
            "%s excludes %d samples from %s through %s UTC: %s",
            spec["id"],
            count,
            start,
            end,
            exclusion["reason"],
        )
    ds = ds.isel(sample=np.flatnonzero(ok))
    ds["lon"] = (ds.lon + 180) % 360 - 180
    if not ds.sizes["sample"]:
        raise ValueError(
            f"No valid observation positions/depths for {spec['id']}"
        )
    ds["sigma0"] = (
        "sample",
        density(
            ds.salt.values,
            ds.temp.values,
            ds.depth.values,
            ds.lon.values,
            ds.lat.values,
        ),
    )
    ds.attrs.update(
        mission=spec["id"],
        source_kind="original_resolution",
        qc="Input QC retained; flag==1 where available; no additional despiking",
        explicit_exclusions=json.dumps(exclusions, sort_keys=True),
    )
    return ds


def grouped_stat(
    values: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized raw-sample mean, population std and finite count in bins."""
    good = (labels >= 0) & np.isfinite(values)
    ncell = int(np.prod(shape))
    n = np.bincount(labels[good], minlength=ncell)
    total = np.bincount(labels[good], weights=values[good], minlength=ncell)
    mean = np.divide(total, n, out=np.full(ncell, np.nan), where=n >= minimum)
    residual = values[good] - mean[labels[good]]
    ss = np.bincount(labels[good], weights=residual**2, minlength=ncell)
    std = np.sqrt(
        np.divide(ss, n, out=np.full(ncell, np.nan), where=n >= minimum)
    )
    return mean.reshape(shape), std.reshape(shape), n.reshape(shape)


def bin_original(
    raw: xr.Dataset,
    depth: np.ndarray,
    centres: np.ndarray,
    settings: dict[str, Any],
) -> xr.Dataset:
    """Raw glider centre-bin averages and paired T/S centroid track, ±3 hours.

    Each cell's model location/time is the centroid of finite paired T/S samples.
    T/S and density means use that same paired support. Empty cells remain empty.
    """
    width = float(settings["window_hours"]) * 3600
    t = raw.time.values.astype("datetime64[ns]").astype(np.int64) / 1e9
    c = centres.astype("datetime64[ns]").astype(np.int64) / 1e9
    iz = (
        np.searchsorted(depth_edges(depth), raw.depth.values, side="right") - 1
    )
    # nearest centre with half-open boundaries; model-time mode can have gaps.
    it = np.searchsorted(c, t)
    it = np.minimum(it, len(c) - 1)
    left = np.maximum(it - 1, 0)
    it = np.where(abs(t - c[left]) < abs(t - c[it]), left, it)
    valid = (
        (t >= c[it] - width / 2)
        & (t < c[it] + width / 2)
        & (iz >= 0)
        & (iz < len(depth))
        & np.isfinite(raw.temp.values)
        & np.isfinite(raw.salt.values)
    )
    labels = np.where(valid, it * len(depth) + iz, -1)
    shape = (len(c), len(depth))
    ds = xr.Dataset(coords={"time": centres, "depth": depth})
    for var, output in (
        ("temp", "gtemp"),
        ("salt", "gsalt"),
        ("sigma0", "gsigma0"),
    ):
        mean, std, n = grouped_stat(
            raw[var].values, labels, shape, settings["min_bin_count"]
        )
        ds[output], ds[output + "_within_std"] = (("time", "depth"), mean), (
            ("time", "depth"),
            std,
        )
        ds[output + "_n"] = (("time", "depth"), n.astype(np.int32))
    for var in ("lon", "lat"):
        mean, _, _ = grouped_stat(
            raw[var].values, labels, shape, settings["min_bin_count"]
        )
        ds[var + "_sample"] = (("time", "depth"), mean)
    origin = c[0]
    offset, _, _ = grouped_stat(
        t - origin, labels, shape, settings["min_bin_count"]
    )
    sample_time = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    good = np.isfinite(offset)
    sample_time[good] = ((offset[good] + origin) * 1e9).astype(
        "datetime64[ns]"
    )
    ds["sample_time"] = (("time", "depth"), sample_time)
    # Preserve completely unsupported windows rather than bridging real gaps.
    ds["depth_bounds"] = (
        ("depth", "bounds"),
        np.column_stack((depth_edges(depth)[:-1], depth_edges(depth)[1:])),
    )
    ds.attrs.update(raw.attrs)
    ds.attrs["observation_method"] = (
        "Paired raw T/S sample mean within midpoint depth bins and centred time window"
    )
    return ds


def bin_stats(
    paths: list[Path],
    spec: dict[str, Any],
    depth: np.ndarray,
    settings: dict[str, Any],
) -> xr.Dataset:
    """Explicit approximate input mode: equal weight per existing depth-bin mean.

    No artificial raw sample counts, within-bin std or density-of-raw-samples are
    inferred from the incomplete 6hrstats sufficient statistics.
    """
    if spec.get("exclude_intervals"):
        raise ValueError(
            "Sample exclusions require original-resolution data; "
            "use stats_policy=raw_companion instead of center_mean"
        )
    with xr.open_mfdataset(paths, combine="by_coords") as source:
        src = source[["gtemp", "gsalt", "lat", "lon"]].load()
    for key, op in (("start", np.greater_equal), ("end", np.less_equal)):
        if spec.get(key):
            src = src.isel(
                time=np.flatnonzero(
                    op(src.time.values, np.datetime64(spec[key]))
                )
            )
    ds = xr.Dataset(coords={"time": src.time.values, "depth": depth})
    edges = depth_edges(depth)
    mask = np.isfinite(src.gtemp.values) & np.isfinite(src.gsalt.values)
    for var in ("gtemp", "gsalt"):
        array = np.full((src.sizes["time"], len(depth)), np.nan)
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            zmask = (src.depth.values >= max(lo, settings["min_depth_m"])) & (
                src.depth.values < hi
            )
            data = np.where(mask[:, zmask], src[var].values[:, zmask], np.nan)
            count = np.isfinite(data).sum(axis=1)
            array[:, j] = np.divide(
                np.nansum(data, axis=1),
                count,
                out=np.full(count.shape, np.nan),
                where=count > 0,
            )
        ds[var] = (("time", "depth"), array)
    for var in ("lat", "lon"):
        ds[var + "_sample"] = (
            ("time", "depth"),
            np.broadcast_to(src[var].values[:, None], ds.gtemp.shape),
        )
    ds["sample_time"] = (
        ("time", "depth"),
        np.broadcast_to(src.time.values[:, None], ds.gtemp.shape),
    )
    ds["gsigma0"] = (
        ("time", "depth"),
        density(
            ds.gsalt.values,
            ds.gtemp.values,
            depth[None, :],
            ds.lon_sample.values,
            ds.lat_sample.values,
        ),
    )
    ds["depth_bounds"] = (
        ("depth", "bounds"),
        np.column_stack((edges[:-1], edges[1:])),
    )
    ds.attrs.update(
        mission=spec["id"],
        source_kind="6hrstats_center_mean",
        observation_method="Unweighted mean of source depth-bin means; NOT raw-sample bin average",
        density_method="sigma0 of remapped mean T/S; raw density mean unavailable",
    )
    LOGGER.warning(
        "%s uses approximate centre-mean remapping of 6hrstats", spec["id"]
    )
    return ds
