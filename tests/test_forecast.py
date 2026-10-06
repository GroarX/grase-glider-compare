"""Forecast tests distinguish initialization, lead time and valid UTC."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grase_glider_compare.forecast import prepare_forecast
from grase_glider_compare.collocate import extract
from test_integration import make_case

logging.basicConfig(level=logging.INFO)


def forecast_case(root: Path) -> dict[str, Any]:
    """Two forecast variants of one generic family, no four-model assumption."""
    config = make_case(root)
    config["models"] = config["models"][:2]
    for spec in config["models"]:
        path = Path(spec["paths"])
        with xr.open_dataset(path) as source:
            ds = source.load()
        init = np.datetime64("2025-04-01T00", "ns")
        leads = (ds.time.values - init) / np.timedelta64(1, "h")
        ds = ds.rename({"time": "lead"}).assign_coords(lead=leads)
        ds = ds.expand_dims(
            forecast_reference_time=[
                init,
                np.datetime64("2025-04-02T00", "ns"),
            ]
        )
        forecast_path = root / ("forecast_" + spec["variant"] + ".nc")
        ds.to_netcdf(forecast_path)
        spec.update(
            id="FORECAST_" + spec["variant"],
            family="FORECAST",
            paths=str(forecast_path),
            forecast={
                "initialization_utc": str(init),
                "init_coordinate": "forecast_reference_time",
                "lead_coordinate": "lead",
                "lead_units": "hours",
            },
        )
        spec["variables"]["time"] = "valid_time"
    return config


def test_fixed_cycle_valid_utc(tmp_path: Path) -> None:
    config = forecast_case(tmp_path / "inputs")
    extract(config, tmp_path, tmp_path / "run")
    for path in (tmp_path / "run/cache").glob("*.nc"):
        with xr.open_dataset(path) as ds:
            assert ds.attrs["experiment"] == "forecast"
            assert ds.attrs["forecast_initialization_utc"].startswith(
                "2025-04-01"
            )
            expected = (
                ds.sample_time.values - np.datetime64("2025-04-01")
            ) / np.timedelta64(1, "h")
            np.testing.assert_allclose(
                ds.forecast_lead_hours.values, expected, equal_nan=True
            )
            assert np.nanmin(ds.forecast_lead_hours) >= 0
            assert ds.paired_valid.sum() > 0
    assert len(list((tmp_path / "run/cache").glob("*.nc"))) == 4


def test_negative_forecast_lead_rejected() -> None:
    ds = xr.Dataset(coords={"lead": [-1.0, 0.0, 1.0]})
    spec = {
        "id": "forecast",
        "variables": {"time": "valid_time"},
        "forecast": {
            "initialization_utc": "2025-01-01",
            "lead_coordinate": "lead",
            "lead_units": "hours",
        },
    }
    with pytest.raises(ValueError, match="precede"):
        prepare_forecast(ds, spec)
