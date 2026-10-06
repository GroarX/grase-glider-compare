"""Analytic scientific tests, independent of the production observations."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from grase_glider_compare.collocate import sample_model
from grase_glider_compare.core import depth_edges, density, insitu, metrics
from grase_glider_compare.inputs import bin_original, observation_paths

logging.basicConfig(level=logging.INFO)


def test_edges_and_average() -> None:
    """Half-open bins, sparse cells and real time gaps have known results."""
    z = np.array([10.0, 20.0, 40.0])
    np.testing.assert_equal(depth_edges(z), [10.0, 15.0, 30.0, 40.0])
    t = np.array(
        ["2025-01-01T00", "2025-01-01T06", "2025-01-01T12"],
        dtype="datetime64[ns]",
    )
    raw = xr.Dataset(
        {
            k: ("sample", v)
            for k, v in {
                "time": np.repeat(t[0], 6),
                "depth": [10.0, 14.0, 15.0, 20.0, 40.0, 40.0],
                "temp": [2.0, 4.0, 6.0, 8.0, 100.0, 100.0],
                "salt": [35.0] * 6,
                "sigma0": [20.0] * 6,
                "lat": [25.0] * 6,
                "lon": [-85.0] * 6,
            }.items()
        }
    )
    out = bin_original(raw, z, t, {"window_hours": 6, "min_bin_count": 2})
    np.testing.assert_allclose(out.gtemp.values[0, :2], [3.0, 7.0])
    assert np.isnan(out.gtemp.values[0, 2])
    assert np.isnan(out.gtemp.values[1:]).all()
    assert out.gtemp_n.values[0, 0] == 2
    np.testing.assert_allclose(out.gtemp_within_std.values[0, 0], 1.0)


def analytic_model() -> xr.Dataset:
    """Field affine in time/latitude/longitude; interpolation must be exact."""
    t = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")
    z, y, x = (
        np.array([10.0, 30.0]),
        np.array([24.0, 26.0]),
        np.array([-86.0, -84.0]),
    )
    a = (
        np.arange(2)[:, None, None, None] * 2
        + z[None, :, None, None] * 0.1
        + y[None, None, :, None] * 0.2
        + x[None, None, None, :] * 0.3
    )
    return xr.Dataset(
        {
            "temp": (("time", "depth", "lat", "lon"), a),
            "salt": (("time", "depth", "lat", "lon"), np.full(a.shape, 35.0)),
        },
        coords={"time": t, "depth": z, "lat": y, "lon": x},
        attrs={"model": "analytic"},
    )


def test_space_time_interpolation() -> None:
    model = analytic_model()
    obs = xr.Dataset(
        {
            "gtemp": (("time", "depth"), np.ones((1, 2))),
            "lon_sample": (("time", "depth"), [[-85.0, -85.0]]),
            "lat_sample": (("time", "depth"), [[25.0, 25.0]]),
            "sample_time": (
                ("time", "depth"),
                np.full((1, 2), np.datetime64("2025-01-01T12", "ns")),
            ),
        },
        coords={
            "time": [np.datetime64("2025-01-01T12", "ns")],
            "depth": [10.0, 30.0],
        },
    )
    settings = {"sampling": "linear", "max_time_gap_hours": 48}
    t, s = sample_model(model, obs, settings)
    np.testing.assert_allclose(t, [[1 + 1 + 5 - 25.5, 1 + 3 + 5 - 25.5]])
    np.testing.assert_allclose(s, 35.0)
    model["temp"][0, 0, 0, 0] = np.nan
    t, _ = sample_model(model, obs, settings)
    assert np.isnan(t[0, 0])  # no wet-corner renormalization
    obs["lon_sample"][:] = -100.0
    t, _ = sample_model(model, obs, settings)
    assert np.isnan(t).all()  # no geographic extrapolation


def test_time_gap_and_exact_endpoint() -> None:
    model = analytic_model()
    obs = xr.Dataset(
        {
            "gtemp": (("time", "depth"), np.ones((1, 2))),
            "lon_sample": (("time", "depth"), [[-86.0, -86.0]]),
            "lat_sample": (("time", "depth"), [[24.0, 24.0]]),
            "sample_time": (
                ("time", "depth"),
                np.full((1, 2), np.datetime64("2025-01-01T12", "ns")),
            ),
        },
        coords={
            "time": [np.datetime64("2025-01-01", "ns")],
            "depth": [10.0, 30.0],
        },
    )
    settings = {"sampling": "linear", "max_time_gap_hours": 6}
    t, _ = sample_model(model, obs, settings)
    assert np.isnan(t).all()
    obs["sample_time"][:] = np.datetime64("2025-01-01", "ns")
    t, _ = sample_model(model, obs, settings)
    np.testing.assert_allclose(t, model.temp.values[:1, :, 0, 0])


def test_metrics_and_teos10() -> None:
    g = np.array([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]])
    stats = metrics(g, g + 2)
    np.testing.assert_allclose(stats["rmsd"], 2.0)
    np.testing.assert_allclose(stats["bias"], 2.0)
    assert np.isnan(stats["nrmsd_obs_std"][1])
    assert np.isnan(stats["r"][1])
    pt = np.array([10.0, 10.0])
    sp = np.array([35.0, 35.0])
    z = np.array([0.0, 1000.0])
    t = insitu(pt, sp, z, -85.0, 25.0)
    np.testing.assert_allclose(t[0], 10.0, atol=1e-10)
    assert t[1] > t[0]
    sigma = density(sp, t, z, -85.0, 25.0)
    assert np.all((sigma > 26) & (sigma < 28))


def test_stats_input_needs_exact_support(tmp_path: Path) -> None:
    p = tmp_path / "mission_6hrstats.nc"
    xr.Dataset(
        {"gtemp": (("time", "depth"), [[10.0, 9.0]])},
        coords={"time": [0.0], "depth": [10.0, 20.0]},
    ).to_netcdf(p)
    with pytest.raises(ValueError, match="Exact rebinning"):
        observation_paths(
            {"paths": str(p), "kind": "6hrstats"}, tmp_path, "raw_companion"
        )
    paths, kind = observation_paths(
        {"paths": str(p), "kind": "6hrstats"}, tmp_path, "center_mean"
    )
    assert paths == [p] and kind == "6hrstats"


def test_stats_center_mean_is_explicit_approximation(tmp_path: Path) -> None:
    """Source-bin means get equal weights, and raw counts are not invented."""
    from grase_glider_compare.inputs import bin_stats

    path = tmp_path / "sample_6hrstats.nc"
    ds = xr.Dataset(
        {
            "gtemp": (("time", "depth"), [[1.0, 3.0, 5.0, 7.0, 9.0, 20.0]]),
            "gsalt": (("time", "depth"), [[35.0] * 6]),
            "lat": ("time", [25.0]),
            "lon": ("time", [-85.0]),
        },
        coords={
            "time": [np.datetime64("2025-01-01", "ns")],
            "depth": [10.0, 12.0, 14.0, 20.0, 26.0, 50.0],
        },
    )
    ds.to_netcdf(path)
    out = bin_stats(
        [path],
        {"id": "sample"},
        np.array([10.0, 20.0, 40.0]),
        {"min_depth_m": 10},
    )
    np.testing.assert_allclose(out.gtemp.values[0, :2], [3.0, 8.0])
    assert np.isnan(out.gtemp.values[0, 2])
    assert "gtemp_n" not in out
    assert out.attrs["source_kind"] == "6hrstats_center_mean"


def test_pooling_preserves_weights_and_simultaneous_missions() -> None:
    """Pool four pairs, not the unweighted average of two mission means."""
    from grase_glider_compare.figures import concatenate, paired
    from grase_glider_compare.core import moments

    parts = []
    for mission, values in (("long", [1.0, 1.0, 1.0]), ("short", [5.0])):
        n = len(values)
        times = np.datetime64("2025-01-01T00", "ns") + np.arange(
            n
        ) * np.timedelta64(6, "h")
        ds = xr.Dataset(
            {
                "gtemp": (("time", "depth"), np.array(values)[:, None]),
                "temp": (("time", "depth"), np.array(values)[:, None]),
                "paired_valid": (
                    ("time", "depth"),
                    np.ones((n, 1), dtype=np.int8),
                ),
                "mission_id": ("time", [mission] * n),
            },
            coords={"time": times, "depth": [20.0]},
        )
        parts.append(ds)
    joined = concatenate(parts)
    assert joined.sizes["time"] == 4
    assert len(np.unique(joined.time)) == 3
    assert set(joined.mission_id.values) == {"long", "short"}
    mean, _, count = moments(paired(joined, "temp")[0])
    np.testing.assert_allclose(mean, [2.0])
    np.testing.assert_equal(count, [4])


def test_explicit_observation_exclusion(tmp_path: Path) -> None:
    """An inclusive recorded interval removes only its exact samples."""
    from grase_glider_compare.inputs import read_original

    path = tmp_path / "original.nc"
    times = np.array(
        [
            "2025-09-02T11:41:20",
            "2025-09-02T11:41:21",
            "2025-09-02T13:12:52",
            "2025-09-02T13:12:53",
        ],
        dtype="datetime64[ns]",
    )
    xr.Dataset(
        {
            "time": ("sample", times),
            "temperature": ("sample", [25.0, 5.0, 5.0, 25.0]),
            "salinity": ("sample", [36.0, 61.0, 61.0, 36.0]),
            "depth": ("sample", [56.0] * 4),
            "lat": ("sample", [25.0] * 4),
            "lon": ("sample", [-85.0] * 4),
        }
    ).to_netcdf(path)
    spec = {"id": "test"}
    settings = {"min_depth_m": 10, "max_depth_m": 1000}
    assert read_original([path], spec, settings).sizes["sample"] == 4
    spec["exclude_intervals"] = [
        {
            "start_utc": "2025-09-02T11:41:21",
            "end_utc": "2025-09-02T13:12:52",
            "reason": "Synthetic sustained anomalous flat values",
            "expected_samples": 2,
        }
    ]
    out = read_original([path], spec, settings)
    np.testing.assert_equal(out.time.values, times[[0, 3]])
    assert '"excluded_samples": 2' in out.attrs["explicit_exclusions"]
    spec["exclude_intervals"][0]["expected_samples"] = 3
    with pytest.raises(ValueError, match="expected 3 samples"):
        read_original([path], spec, settings)
