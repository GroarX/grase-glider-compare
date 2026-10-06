"""Command-line entry point; configuration paths are relative to the JSON file."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .collocate import extract

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Validate model/mission/group identities before starting any work."""
    config = json.loads(path.read_text())
    for key in ("models", "missions"):
        ids = [item["id"] for item in config[key]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {key} IDs")
        if any(
            not value
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in value
            )
            for value in ids
        ):
            raise ValueError("Use safe alphanumeric, underscore or hyphen IDs")
    for model in config["models"]:
        if model["variant"] not in ("grace", "nograce", "reference"):
            raise ValueError(
                "Variant must be grace (GrASE), nograce (NoGrASE), or reference"
            )
        if model["id"] != model["family"] + "_" + model["variant"]:
            raise ValueError("Model ID must be <family>_<variant>")
    for family in {m["family"] for m in config["models"]}:
        pair = [m for m in config["models"] if m["family"] == family]
        forecasts = [m.get("forecast") for m in pair]
        if any(forecasts):
            if not all(forecasts):
                raise ValueError(
                    "Do not mix hindcast and forecast variants within one family"
                )
            cycles = {f["initialization_utc"] for f in forecasts}
            if len(cycles) != 1:
                raise ValueError(
                    "GrASE/NoGrASE forecast comparisons require the same initialization UTC"
                )
    missions = {m["id"] for m in config["missions"]}
    for group, names in config.get("groups", {}).items():
        if (
            not names
            or len(names) != len(set(names))
            or not set(names) <= missions
        ):
            raise ValueError(f"Invalid group {group}: {names}")
    defaults = {
        "window_hours": 6,
        "min_depth_m": 10,
        "max_depth_m": 1000,
        "min_bin_count": 2,
        "max_time_gap_hours": 48,
        "sampling": "linear",
        "stats_policy": "raw_companion",
        "seed": 20261005,
    }
    defaults.update(config.get("settings", {}))
    config["settings"] = defaults
    if defaults["window_hours"] != 6:
        raise ValueError(
            "Current UTC window implementation requires window_hours=6"
        )
    if defaults["sampling"] not in ("linear", "legacy_nearest"):
        raise ValueError("Unknown sampling method")
    if defaults["stats_policy"] not in ("raw_companion", "center_mean"):
        raise ValueError("Unknown stats_policy")
    return config


def main() -> None:
    """Dispatch extraction, plotting, verification or complete runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("extract", "plot", "sections", "verify", "run")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--missions", nargs="+")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--skip-individual", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for names, key in ((args.models, "models"), (args.missions, "missions")):
        if names and not set(names) <= {s["id"] for s in config[key]}:
            parser.error(f"Unknown {key}: {names}")
    if args.groups and not set(args.groups) <= set(config.get("groups", {})):
        parser.error("Unknown group")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in ("run", "extract"):
        extract(
            config,
            args.config.resolve().parent,
            args.output,
            args.models,
            args.missions,
        )
    if args.command in ("run", "plot"):
        from .figures import render

        render(
            config,
            args.output,
            args.models,
            args.missions,
            args.groups,
            args.skip_individual,
            args.config.resolve().parent,
        )
    if args.command == "sections":
        from .sections import render_sections

        render_sections(config, args.output)
    if args.command in ("run", "verify"):
        from .verify import verify

        selected = dict(config)
        selected["models"] = [
            m
            for m in config["models"]
            if not args.models or m["id"] in args.models
        ]
        selected["missions"] = [
            m
            for m in config["missions"]
            if not args.missions or m["id"] in args.missions
        ]
        verify(selected, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Hindcast/glider task failed")
        raise
