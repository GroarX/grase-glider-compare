"""Explicit forecast valid-time preparation without mixing forecast cycles."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def prepare_forecast(source: xr.Dataset, spec: dict[str, Any]) -> xr.Dataset:
    """Select one declared initialization and construct/check valid UTC.

    Two supported interfaces: already-decoded valid_time, or numeric/timedelta
    lead coordinates for a fixed initialization. Multiple initialization axes
    must be selected explicitly with init_coordinate and initialization_utc.
    """
    options = spec.get("forecast")
    if not options:
        return source
    initialization = np.datetime64(options["initialization_utc"], "ns")
    if np.isnat(initialization):
        raise ValueError("Forecast initialization_utc must be a real UTC date")
    init_coord = options.get("init_coordinate")
    if init_coord:
        if init_coord not in source:
            raise ValueError(
                f"Missing forecast initialization coordinate {init_coord}"
            )
        init = source[init_coord]
        if init.ndim == 0:
            if np.datetime64(init.values, "ns") != initialization:
                raise ValueError(
                    "File initialization does not match initialization_utc"
                )
        else:
            source = source.sel({init_coord: initialization}, drop=True)
    time_name = spec["variables"]["time"]
    lead_name = options.get("lead_coordinate")
    if lead_name:
        lead = source[lead_name]
        if lead.ndim != 1:
            raise ValueError(
                "Select one forecast initialization before collocation"
            )
        if np.issubdtype(lead.dtype, np.timedelta64):
            delta = lead.values.astype("timedelta64[ns]")
        else:
            unit = options.get("lead_units")
            scales = {"hours": 3600.0, "days": 86400.0, "seconds": 1.0}
            if unit not in scales:
                raise ValueError(
                    "Numeric forecast lead_units must be hours, days, or seconds"
                )
            delta = (lead.values.astype(float) * scales[unit] * 1e9).astype(
                "timedelta64[ns]"
            )
        valid = initialization + delta
        dimension = lead.dims[0]
        source = source.assign_coords({time_name: (dimension, valid)})
        if dimension != time_name:
            source = source.swap_dims({dimension: time_name})
    else:
        valid = source[time_name].values
        if not np.issubdtype(valid.dtype, np.datetime64):
            raise ValueError(
                "Forecast time must be valid UTC, or configure lead_coordinate"
            )
    if np.any(valid < initialization):
        raise ValueError(
            "Forecast valid times precede the selected initialization"
        )
    source.attrs = dict(source.attrs)
    source.attrs.update(
        forecast_initialization_utc=str(initialization),
        forecast_time_definition="valid_time = initialization + lead",
        forecast_cycle_policy="one explicitly selected initialization",
    )
    LOGGER.info(
        "%s forecast initialization=%s; valid UTC %s to %s",
        spec["id"],
        initialization,
        valid.min(),
        valid.max(),
    )
    return source
