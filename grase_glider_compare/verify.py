"""Audit cache coverage, units, finite pairs, output integrity and actual PNG dpi."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from PIL import Image

LOGGER = logging.getLogger(__name__)


def verify(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Fail visibly if a requested mission/model or rendered figure is missing."""
    import importlib.metadata

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "xarray",
            "netCDF4",
            "dask",
            "matplotlib",
            "cmocean",
            "gsw",
            "Pillow",
        )
    }
    (output / "software_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n"
    )
    report: dict[str, Any] = {"caches": [], "figures": [], "errors": []}
    for model in config["models"]:
        for mission in config["missions"]:
            path = output / "cache" / f"{model['id']}_{mission['id']}_bins.nc"
            if not path.exists():
                report["errors"].append(f"Missing cache: {path.name}")
                continue
            with xr.open_dataset(path) as ds:
                mask = np.ones(ds.gtemp.shape, dtype=bool)
                for var in (
                    "gtemp",
                    "gsalt",
                    "gsigma0",
                    "temp",
                    "salt",
                    "sigma0",
                ):
                    mask &= np.isfinite(ds[var].values)
                if not np.array_equal(
                    mask, ds.paired_valid.values.astype(bool)
                ):
                    report["errors"].append(
                        f"Invalid paired mask: {path.name}"
                    )
                if not mask.any():
                    report["errors"].append(f"No paired samples: {path.name}")
                if not np.all(
                    np.diff(ds.time.values) > np.timedelta64(0, "ns")
                ):
                    report["errors"].append(
                        f"Non-increasing time: {path.name}"
                    )
                if not np.all(np.diff(ds.depth.values) > 0):
                    report["errors"].append(
                        f"Non-increasing depth: {path.name}"
                    )
                physical_ranges = {}
                for var, low, high in (
                    ("gtemp", -5, 50),
                    ("temp", -5, 50),
                    ("gsalt", 0, 50),
                    ("salt", 0, 50),
                    ("gsigma0", -5, 50),
                    ("sigma0", -5, 50),
                ):
                    values = ds[var].values[mask]
                    physical_ranges[var] = (
                        [float(values.min()), float(values.max())]
                        if values.size
                        else [None, None]
                    )
                    if np.any((values < low) | (values > high)):
                        report["errors"].append(
                            f"Physical/unit sanity check failed: {path.name} {var}"
                        )
                report["caches"].append(
                    {
                        "file": path.name,
                        "time": ds.sizes["time"],
                        "depth": ds.sizes["depth"],
                        "paired_cells": int(mask.sum()),
                        "start_utc": str(ds.time.values[0]),
                        "end_utc": str(ds.time.values[-1]),
                        "source_kind": ds.attrs["source_kind"],
                        "sampling": ds.attrs["sampling"],
                        "fingerprint": ds.attrs["fingerprint"],
                        "paired_value_ranges": physical_ranges,
                    }
                )
    manifest = output / "figure_manifest.json"
    if not manifest.exists():
        report["errors"].append("Missing figure manifest")
    else:
        records = json.loads(manifest.read_text())
        if not records:
            report["errors"].append("Empty figure manifest")
        resolved = [str(Path(item["file"]).resolve()) for item in records]
        if len(resolved) != len(set(resolved)):
            report["errors"].append("Duplicate entries in figure manifest")
        for item in records:
            path = Path(item["file"])
            if not path.exists() or not path.stat().st_size:
                report["errors"].append(f"Missing figure {path}")
                continue
            with Image.open(path) as im:
                im.load()
                dpi = im.info.get("dpi", (0, 0))
                if abs(dpi[0] - 300) > 0.1 or abs(dpi[1] - 300) > 0.1:
                    report["errors"].append(
                        f"Invalid resolution {path}: {dpi}"
                    )
                if dpi[0] > 0 and abs(im.width / dpi[0] * 25.4 - 190) > 2:
                    report["errors"].append(
                        f"Unexpected publication width {path}: {im.width/dpi[0]*25.4}"
                    )
                report["figures"].append(
                    {"file": str(path), "pixels": im.size, "dpi": dpi}
                )
    report["complete"] = not report["errors"]
    (output / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    LOGGER.info(
        "Verified %d caches and %d figures; errors=%d",
        len(report["caches"]),
        len(report["figures"]),
        len(report["errors"]),
    )
    if report["errors"]:
        raise RuntimeError("; ".join(report["errors"][:10]))
    return report
