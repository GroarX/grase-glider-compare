"""Exercise standalone products without assigning a GrASE assimilation label."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import xarray as xr

from grase_glider_compare.cli import load_config
from grase_glider_compare.collocate import extract
from grase_glider_compare.figures import render
from grase_glider_compare.verify import verify
from test_integration import make_case

logging.basicConfig(level=logging.INFO)


def test_reference_products_extract_and_render(tmp_path: Path) -> None:
    """Check mapped variables, external labels and complete reference figures."""
    config = make_case(tmp_path / "inputs")
    config["models"] = config["models"][:2]
    config["missions"] = config["missions"][:1]
    config["groups"] = {}
    mappings = [
        ("ESPC", {"temp": "water_temp", "salt": "salinity"}),
        (
            "GLORYS",
            {"temp": "thetao", "salt": "so", "lat": "latitude", "lon": "longitude"},
        ),
    ]
    for spec, (family, names) in zip(config["models"], mappings):
        with xr.open_dataset(spec["paths"]) as source:
            renamed = source.load().rename(names)
        path = tmp_path / "inputs" / (family + ".nc")
        renamed.to_netcdf(path)
        spec.update(
            id=family + "_reference",
            family=family,
            variant="reference",
            paths=str(path),
        )
        spec["variables"].update(names)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    config = load_config(config_path)
    output = tmp_path / "results"
    extract(config, tmp_path, output)
    for path in (output / "cache").glob("*.nc"):
        with xr.open_dataset(path) as ds:
            assert ds.attrs["variant"] == "reference"
            assert ds.paired_valid.sum() > 100
    render(config, output, None, None, None, False, tmp_path)
    verify(config, output)
    figures = list((output / "figures").rglob("*.png"))
    assert len(figures) == 8
    assert any("TS_profiles_reference" in p.name for p in figures)
    assert any("TS25panels_reference" in p.name for p in figures)
    assert any("state_histograms_reference" in p.name for p in figures)
    assert sum("nrmsd" in p.name for p in figures) == 2
    assert not any("diff_hist_depth50m" in p.name for p in figures)
    assert not any("_nograce" in p.name or "_grace" in p.name for p in figures)
