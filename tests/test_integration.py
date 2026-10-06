"""Small end-to-end original and 6hrstats-input tests on a known model field."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from grase_glider_compare.cli import load_config
from grase_glider_compare.collocate import extract
from grase_glider_compare.inputs import open_model

logging.basicConfig(level=logging.INFO)


def make_case(root: Path) -> dict[str, Any]:
    """Synthetic five-day hindcast; two missions with different observations."""
    root.mkdir(parents=True, exist_ok=True)
    depth = np.array(
        [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
    )
    time = np.arange(
        np.datetime64("2025-04-01"),
        np.datetime64("2025-04-07"),
        np.timedelta64(1, "D"),
    ).astype("datetime64[ns]")
    lat = np.array([24.0, 25.0, 26.0])
    lon = np.array([-86.0, -85.0, -84.0])
    temp = (
        28
        - depth[None, :, None, None] * 0.02
        + np.arange(len(time))[:, None, None, None] * 0.1
        + np.zeros((len(time), len(depth), 3, 3))
    )
    salt = 35.5 + depth[None, :, None, None] * 0.001 + np.zeros_like(temp)
    missions = []
    for i in range(2):
        # Bin-centre depths +/- small offsets and sufficient samples per window.
        centres = np.arange(
            np.datetime64("2025-04-01T06"),
            np.datetime64("2025-04-05T18"),
            np.timedelta64(6, "h"),
        ).astype("datetime64[ns]")
        z = np.tile(np.repeat(depth[:-1] + 1, 4), len(centres))
        tt = np.repeat(centres, (len(depth) - 1) * 4)
        t = 28 - z * 0.02 + np.sin(np.arange(len(z)) * 0.08) * 0.3 + i * 0.3
        sp = 35.5 + z * 0.001 + np.cos(np.arange(len(z)) * 0.08) * 0.03
        m = f"mission_{i}"
        raw = xr.Dataset(
            {
                k: ("time", v)
                for k, v in {
                    "depth": z,
                    "temperature": t,
                    "salinity": sp,
                    "lat": np.full(z.size, 24.8 + i * 0.3),
                    "lon": np.full(z.size, -85.2 + i * 0.3),
                }.items()
            },
            coords={"time": tt},
        )
        raw.to_netcdf(root / f"{m}_original_resolution.nc")
        missions.append(
            dict(
                id=m,
                paths=str(root / f"{m}_original_resolution.nc"),
                kind="original",
            )
        )
    models = []
    for f, family in enumerate(
        ("HYCOM-FSU", "ROMS-NCSU", "MITgcm-UCSD", "ROMS-CICESE")
    ):
        for j, variant in enumerate(("nograce", "grace")):
            ds = xr.Dataset(
                {
                    "temp": (
                        ("time", "depth", "lat", "lon"),
                        temp + 0.2 * f - 0.1 * j,
                    ),
                    "salt": (
                        ("time", "depth", "lat", "lon"),
                        salt + 0.02 * f - 0.01 * j,
                    ),
                },
                coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
            )
            names = dict(
                time="time",
                depth="depth",
                lat="lat",
                lon="lon",
                temp="temp",
                salt="salt",
            )
            if family == "ROMS-NCSU":
                ds = ds.rename({"lat": "y", "lon": "x"})
                xx, yy = np.meshgrid(lon, lat)
                ds["latitude"] = (("y", "x"), yy)
                ds["longitude"] = (("y", "x"), xx)
                names.update(lat="latitude", lon="longitude")
            spec = dict(
                id=family + "_" + variant,
                family=family,
                variant=variant,
                variables=names,
                temperature_kind="in_situ",
            )
            if family == "MITgcm-UCSD":
                dn = (
                    time.astype("datetime64[s]").astype(float) / 86400 + 719529
                )
                ds = ds.assign_coords(time=dn)
                spec["time_encoding"] = "matlab_datenum"
            path = root / (spec["id"] + ".nc")
            ds.to_netcdf(path)
            spec["paths"] = str(path)
            models.append(spec)
    return dict(
        models=models,
        missions=missions,
        groups={"both_missions": [m["id"] for m in missions]},
        settings={
            "window_hours": 6,
            "min_depth_m": 10,
            "max_depth_m": 1000,
            "min_bin_count": 2,
            "max_time_gap_hours": 48,
            "sampling": "linear",
            "stats_policy": "raw_companion",
            "seed": 20261005,
        },
    )


def test_all_adapters_and_16_pairs(tmp_path: Path) -> None:
    config = make_case(tmp_path / "inputs")
    extract(config, tmp_path, tmp_path / "outputs")
    paths = list((tmp_path / "outputs/cache").glob("*.nc"))
    assert len(paths) == 16
    for path in paths:
        with xr.open_dataset(path) as ds:
            assert ds.paired_valid.sum() > 100
            assert ds.attrs["difference_sign"] == "model minus glider"
            assert ds.attrs["experiment"] == "model_output"
            assert ds.temp.dims == ("time", "depth")
            assert ds.gtemp_n.max() == 4
            assert np.isnan(ds.gtemp.values[:, -1]).all()
    # Exact same input/setting/code fingerprint must reuse, not overwrite.
    before = paths[0].stat().st_mtime_ns
    extract(
        config,
        tmp_path,
        tmp_path / "outputs",
        models=[config["models"][0]["id"]],
    )
    assert paths[0].stat().st_mtime_ns == before


def test_6hrstats_companion_path(tmp_path: Path) -> None:
    config = make_case(tmp_path / "inputs")
    config["models"] = config["models"][:1]
    config["missions"] = config["missions"][:1]
    mission = config["missions"][0]
    path = Path(mission["paths"].replace("_original_resolution", "_6hrstats"))
    xr.Dataset(
        {"gtemp": (("time", "depth"), [[1.0, 2.0]])},
        coords={"time": [0.0], "depth": [10.0, 20.0]},
    ).to_netcdf(path)
    mission["paths"] = str(path)
    mission["kind"] = "6hrstats"
    extract(config, tmp_path, tmp_path / "outputs")
    with xr.open_dataset(
        next((tmp_path / "outputs/cache").glob("*.nc"))
    ) as ds:
        assert ds.attrs["source_kind"] == "original_resolution"
        assert "_6hrstats.nc" in ds.attrs["requested_glider_files"]
        assert "_original_resolution.nc" in ds.attrs["source_files"]
