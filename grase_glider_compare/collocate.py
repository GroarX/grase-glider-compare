"""Bounded-memory space/time interpolation and provenance-bearing caches."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .core import atomic_netcdf, density, insitu, signature
from .inputs import (
    bin_original,
    bin_stats,
    expand,
    observation_paths,
    open_model,
    read_original,
)

LOGGER = logging.getLogger(__name__)


def bracket(axis: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return bracketing indices/fractions; exact endpoints have zero weight."""
    upper = np.clip(
        np.searchsorted(axis, target, side="right"), 1, len(axis) - 1
    )
    lower = upper - 1
    weight = (target - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


def sample_model(
    model: xr.Dataset, obs: xr.Dataset, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear horizontal / linear temporal interpolation on native z levels.

    Read one small bounding rectangle per used model timestamp. Require every
    nonzero-weight corner to be finite; never renormalize around land/missing
    cells, extrapolate, or interpolate across excessive time gaps.
    """
    shape = obs.gtemp.shape
    lon, lat = obs.lon_sample.values.ravel(), obs.lat_sample.values.ravel()
    query = obs.sample_time.values
    if settings["sampling"] == "legacy_nearest":
        query = np.broadcast_to(obs.time.values[:, None], shape)
        # Legacy representative location is a mean over depth-cell centroids;
        # documented as a compatibility diagnostic, not bitwise historical QC.
    tq = query.astype("datetime64[ns]").astype(np.int64).ravel() / 1e9
    times = model.time.values.astype("datetime64[ns]").astype(np.int64) / 1e9
    yy, xx = model.lat.values, model.lon.values
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & ~np.isnat(query).ravel()
        & (lon >= xx[0])
        & (lon <= xx[-1])
        & (lat >= yy[0])
        & (lat <= yy[-1])
        & (tq >= times[0])
        & (tq <= times[-1])
    )
    i0, i1, wx = bracket(xx, lon)
    j0, j1, wy = bracket(yy, lat)
    t0, t1, wt = bracket(times, tq)
    exact = np.isclose(wt, 0, atol=1e-9) | np.isclose(wt, 1, atol=1e-9)
    valid &= exact | (
        (times[t1] - times[t0]) <= settings["max_time_gap_hours"] * 3600
    )
    if settings["sampling"] == "legacy_nearest":
        i0 = np.where(wx < 0.5, i0, i1)
        j0 = np.where(wy < 0.5, j0, j1)
        i1, j1 = i0.copy(), j0.copy()
        wx, wy = np.zeros_like(wx), np.zeros_like(wy)
        t0 = np.where(wt < 0.5, t0, t1)
        t1, wt = t0.copy(), np.zeros_like(wt)
    k = np.tile(np.arange(shape[1]), shape[0])
    accum = [np.zeros(lon.size), np.zeros(lon.size)]
    bad = [~valid.copy(), ~valid.copy()]
    # Aggregate all reads belonging to one timestamp, avoiding repeated full fields.
    indices = np.unique(np.r_[t0[valid], t1[valid]])
    for count, ti in enumerate(indices):
        active = valid & (
            ((t0 == ti) & ((1 - wt) > 1e-12)) | ((t1 == ti) & (wt > 1e-12))
        )
        ids = np.flatnonzero(active)
        if not ids.size:
            continue
        imin, imax = min(i0[ids].min(), i1[ids].min()), max(
            i0[ids].max(), i1[ids].max()
        )
        jmin, jmax = min(j0[ids].min(), j1[ids].min()), max(
            j0[ids].max(), j1[ids].max()
        )
        block = (
            model[["temp", "salt"]]
            .isel(
                time=int(ti),
                lon=slice(int(imin), int(imax) + 1),
                lat=slice(int(jmin), int(jmax) + 1),
            )
            .load()
        )
        time_weight = np.where(t0[ids] == ti, 1 - wt[ids], 0) + np.where(
            t1[ids] == ti, wt[ids], 0
        )
        for h, var in enumerate(("temp", "salt")):
            field = block[var].values
            for ix, xw in ((i0, 1 - wx), (i1, wx)):
                for jy, yw in ((j0, 1 - wy), (j1, wy)):
                    weights = time_weight * xw[ids] * yw[ids]
                    use = weights > 1e-12
                    values = field[k[ids], jy[ids] - jmin, ix[ids] - imin]
                    bad[h][ids] |= use & ~np.isfinite(values)
                    accum[h][ids] += np.where(
                        use & np.isfinite(values), values * weights, 0
                    )
        if count % 40 == 0:
            LOGGER.info(
                "Interpolating %s: model time slab %d/%d",
                model.attrs["model"],
                count + 1,
                len(indices),
            )
    result = []
    for values, mask in zip(accum, bad):
        values[mask] = np.nan
        result.append(values.reshape(shape))
    return result[0], result[1]


def build_one(
    model: xr.Dataset,
    model_paths: list[Path],
    mspec: dict[str, Any],
    mission: dict[str, Any],
    config: dict[str, Any],
    root: Path,
    output: Path,
) -> Path:
    """Create one new cache or reuse an exactly matching existing cache."""
    settings = config["settings"]
    opaths, kind = observation_paths(mission, root, settings["stats_policy"])
    requested = expand(mission["paths"], root)
    fp = signature(
        model_paths + list(dict.fromkeys(opaths + requested)),
        {"settings": settings, "model": mspec, "mission": mission},
    )
    path = output / "cache" / f"{mspec['id']}_{mission['id']}_bins.nc"
    if path.exists():
        with xr.open_dataset(path) as previous:
            if previous.attrs.get("fingerprint") != fp:
                raise ValueError(
                    f"Stale cache {path}; preserve it and select a new output directory"
                )
        LOGGER.info("Reusing verified cache %s", path)
        return path
    depth = model.depth.values
    if kind == "original":
        raw = read_original(opaths, mission, settings)
        lo = max(
            pd.Timestamp(raw.time.values.min()).floor("6h"),
            pd.Timestamp(model.time.values.min()),
        )
        hi = min(
            pd.Timestamp(raw.time.values.max()).ceil("6h"),
            pd.Timestamp(model.time.values.max()),
        )
        if lo > hi:
            raise ValueError(
                f"No model/glider time overlap: {mspec['id']} {mission['id']}"
            )
        if settings["sampling"] == "legacy_nearest":
            centres = model.time.values[
                (model.time.values >= lo.to_datetime64())
                & (model.time.values <= hi.to_datetime64())
            ]
        else:
            centres = pd.date_range(
                lo.ceil("6h"), hi.floor("6h"), freq="6h"
            ).values
        if not len(centres):
            raise ValueError("No valid bin centre times")
        obs = bin_original(raw, depth, centres, settings)
    else:
        obs = bin_stats(opaths, mission, depth, settings)
        obs = obs.sel(
            time=slice(model.time.values.min(), model.time.values.max())
        )
    temp, salt = sample_model(model, obs, settings)
    if mspec["temperature_kind"] == "potential_0dbar":
        temp = insitu(
            temp,
            salt,
            depth[None, :],
            obs.lon_sample.values,
            obs.lat_sample.values,
        )
    elif mspec["temperature_kind"] != "in_situ":
        raise ValueError("temperature_kind must be potential_0dbar or in_situ")
    obs["temp"], obs["salt"] = (("time", "depth"), temp), (
        ("time", "depth"),
        salt,
    )
    obs["sigma0"] = (
        ("time", "depth"),
        density(
            salt,
            temp,
            depth[None, :],
            obs.lon_sample.values,
            obs.lat_sample.values,
        ),
    )
    # Common support within each model/glider comparison across T/S/sigma0.
    paired = np.ones(obs.gtemp.shape, dtype=bool)
    for var in ("gtemp", "gsalt", "gsigma0", "temp", "salt", "sigma0"):
        paired &= np.isfinite(obs[var].values)
    obs["paired_valid"] = (("time", "depth"), paired.astype(np.int8))
    obs["mission_id"] = ("time", np.repeat(mission["id"], obs.sizes["time"]))
    obs.attrs.update(
        model=mspec["id"],
        family=mspec["family"],
        variant=mspec["variant"],
        fingerprint=fp,
        sampling=settings["sampling"],
        settings=json.dumps(settings, sort_keys=True),
        source_files=json.dumps([str(p) for p in model_paths + opaths]),
        requested_glider_files=json.dumps([str(p) for p in requested]),
        difference_sign="model minus glider",
        time_standard="UTC",
        density_definition="TEOS-10 potential-density anomaly sigma0, reference pressure 0 dbar",
        model_temperature_input=mspec["temperature_kind"],
        model_temperature_output="in_situ",
        pairing="Within each model: joint finite T/S/sigma0; no cross-model common-support assumption",
    )
    if mspec.get("forecast"):
        initialization = np.datetime64(
            mspec["forecast"]["initialization_utc"], "ns"
        )
        obs.attrs.update(
            forecast_initialization_utc=str(initialization),
            experiment="forecast",
            forecast_cycle_policy="fixed initialization",
        )
        lead = (obs.sample_time.values - initialization) / np.timedelta64(
            1, "h"
        )
        obs["forecast_lead_hours"] = (("time", "depth"), lead)
        obs.forecast_lead_hours.attrs["units"] = "hours"
    else:
        obs.attrs["experiment"] = "model_output"
    for var in obs:
        if "temp" in var and not var.endswith("_n"):
            obs[var].attrs["units"] = "degree_Celsius"
        elif "salt" in var and not var.endswith("_n"):
            obs[var].attrs["units"] = "1"
        elif "sigma0" in var and not var.endswith("_n"):
            obs[var].attrs["units"] = "kg m-3"
    obs.depth.attrs.update(units="m", positive="down", bounds="depth_bounds")
    for var in ("lat_sample", "lon_sample"):
        obs[var].attrs["units"] = (
            "degrees_north" if var.startswith("lat") else "degrees_east"
        )
    if not paired.any():
        raise ValueError(
            f"No paired support for {mspec['id']} / {mission['id']}"
        )
    atomic_netcdf(obs, path)
    LOGGER.info(
        "%s: %d joint-finite time-depth pairs", path.name, paired.sum()
    )
    return path


def extract(
    config: dict[str, Any],
    root: Path,
    output: Path,
    models: list[str] | None = None,
    missions: list[str] | None = None,
) -> None:
    """Run selected model/mission extraction; suitable for one-model Slurm tasks."""
    for spec in config["models"]:
        if models and spec["id"] not in models:
            continue
        paths = expand(spec["paths"], root)
        with open_model(
            paths, spec, config["settings"]["max_depth_m"]
        ) as model:
            LOGGER.info(
                "%s: %s, UTC %s to %s",
                spec["id"],
                dict(model.sizes),
                model.time.values[0],
                model.time.values[-1],
            )
            for mission in config["missions"]:
                if missions and mission["id"] not in missions:
                    continue
                build_one(model, paths, spec, mission, config, root, output)
