# grase-glider-compare

Compare model outputs with one or more glider missions, including GrASE/NoGrASE pairs, ESPC and GLORYS. Generates T–S diagrams, profile and section comparisons, difference plots, histograms and vertical NRMSD plots.

## Install

Python 3.11 or newer:

```bash
python -m pip install .
```

## Run

Edit file paths and variable names in `configs/model_pair.example.json`, then:

```bash
grase-glider-compare run --config configs/model_pair.example.json --output results/my_comparison
```

The equivalent Python command is `python -m grase_glider_compare run ...`.

| Input | Configuration |
|---|---|
| GrASE/NoGrASE hindcast or forecast with valid UTC | `configs/model_pair.example.json` |
| Forecast initialization + lead time | `configs/forecast_pair.example.json` |
| ESPC/GLORYS as standalone references | `configs/external_models.example.json` |
| Four model pairs and multiple mission groups | `configs/four_models.example.json` |

Model inputs require temperature in °C, practical salinity, physical depth in metres, a separable latitude/longitude grid and valid UTC. Select `potential_0dbar` or `in_situ` to match the temperature definition. Glider inputs are `*_original_resolution.nc` or `*_6hrstats.nc`. Paths are relative to the configuration file. See [input details](docs/inputs.md).

## Complete example

[HYCOM versus all four Yucatan-side missions](examples/hycom_yucatan/) includes the configuration and every figure from a real GrASE/NoGrASE comparison.

```bash
grase-glider-compare run --config examples/hycom_yucatan/config.json --output results/hycom_yucatan
```

All supplied comparison results use `*_original_resolution.nc` glider observations, averaged into six-hour windows and model depth bins by this package. They do not use the existing `*_6hrstats.nc` means.

[Data share](https://drive.google.com/drive/folders/1MJ9wrgEqYmXIEWtJ25PRUQ3-RaIrC-T9?usp=drive_link). Supply the same model and glider files and update their paths to reproduce the example. Data and numerical caches are not included in this repository.

For an existing run, replace `run` with `plot` to redraw its figures. Use a new output folder when inputs or scientific settings change.

Differences are model minus glider. Density is σ₀. Group statistics pool paired samples. See [METHODS.md](METHODS.md) for definitions and [README.txt](README.txt) for plain-text instructions.

## Cite

Ge, X. (2026). grase-glider-compare (v1.2.0). Zenodo. https://doi.org/10.5281/zenodo.23194558

License: MIT.
