"""Re-render only time-depth sections from unchanged scientific caches."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import xarray as xr

from .figures import plot_assembled, plot_four_model, style, VARIANT_LABEL

LOGGER = logging.getLogger(__name__)


def render_sections(config: dict[str, Any], output: Path) -> None:
    """Refresh section families and merge their records into the full manifest."""
    style()
    registry: list[dict[str, Any]] = []
    specs = config["models"]
    mission_ids = [m["id"] for m in config["missions"]]
    datasets: dict[tuple[str, str], xr.Dataset] = {}
    for spec in specs:
        for mission in mission_ids:
            with xr.open_dataset(
                output / "cache" / f"{spec['id']}_{mission}_bins.nc"
            ) as ds:
                datasets[(spec["id"], mission)] = ds.load()
    figures = output / "figures"
    for mission in mission_ids:
        for variant in ("nograce", "grace"):
            ds4 = {
                s["family"]: datasets[(s["id"], mission)]
                for s in specs
                if s["variant"] == variant
            }
            if not ds4:
                continue
            prefix = "four_model" if len(ds4) == 4 else f"{len(ds4)}_model"
            title = f"{mission} | {VARIANT_LABEL[variant]}"
            for mode, tag in (("difference", "compare"), ("model", "state")):
                plot_four_model(
                    ds4,
                    title,
                    figures
                    / mission
                    / f"{prefix}_{tag}_{variant}_{mission}.png",
                    mode,
                    registry,
                )
    for name, missions in config.get("groups", {}).items():
        for spec in specs:
            parts = {m: datasets[(spec["id"], m)] for m in missions}
            plot_assembled(
                parts,
                f"{spec['id']} | {name}",
                figures / name / f"{spec['id']}_{name}_bins.png",
                registry,
            )
    manifest_path = output / "figure_manifest.json"
    previous = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    )
    updated = {str(Path(item["file"]).resolve()) for item in registry}
    records = [
        item
        for item in previous
        if str(Path(item["file"]).resolve()) not in updated
    ] + registry
    manifest_path.write_text(json.dumps(records, indent=2) + "\n")
    LOGGER.info(
        "Updated %d section figures; total manifest %d",
        len(registry),
        len(records),
    )
