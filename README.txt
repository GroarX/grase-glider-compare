grase-glider-compare

Compare model outputs with one or more glider missions, including GrASE/NoGrASE
pairs, ESPC and GLORYS. Generates T-S diagrams, profile and section comparisons,
difference plots, histograms and vertical NRMSD plots.

INSTALL (Python 3.11+)
    python -m pip install .

RUN
Edit paths and variable names in configs/model_pair.example.json, then:
    grase-glider-compare run --config configs/model_pair.example.json --output results/my_comparison

Equivalent command: python -m grase_glider_compare run ...

CONFIGURATIONS
    model_pair.example.json:      GrASE/NoGrASE hindcast or forecast, valid UTC
    forecast_pair.example.json:   forecast initialization plus lead time
    external_models.example.json: ESPC/GLORYS with variant "reference"
    four_models.example.json:     four pairs, multiple missions and groups

Paths are relative to the JSON file. Model fields require temperature in Celsius,
practical salinity, physical depth in metres and a separable latitude/longitude
grid. Set temperature_kind to potential_0dbar or in_situ. Merge separate temperature
and salinity files before input. Time must be valid UTC. If initialization is
specified, use the same one for GrASE and NoGrASE. See docs/inputs.md.

GLIDERS
Use *_original_resolution.nc, or *_6hrstats.nc. Six-hour input defaults to exact
rebinning from its original-resolution companion. To read the six-hour means
directly, set settings.stats_policy to center_mean (approximate rebinning).

COMPLETE REAL-DATA EXAMPLE
HYCOM GrASE/NoGrASE versus all four Yucatan-side missions:
    grase-glider-compare run --config examples/hycom_yucatan/config.json --output results/hycom_yucatan

Edit paths to your copies of the same inputs. All example figures are under
examples/hycom_yucatan/figures/. All supplied comparison results use
*_original_resolution.nc, averaged here into six-hour windows and model depth
bins, rather than the existing *_6hrstats.nc means.

Shared input data:
https://drive.google.com/drive/folders/1MJ9wrgEqYmXIEWtJ25PRUQ3-RaIrC-T9?usp=drive_link
Input data and numerical caches are not included in this repository.

OUTPUTS
    figures/: publication figures and captions
    cache/:   matched NetCDF files
    metrics/: depth-dependent statistics

Replace run with plot to redraw existing caches. Use a new output folder after
changing inputs or scientific settings. Differences are model minus glider;
density is sigma0. Profiles show mean +/-1 standard deviation. Group statistics
pool paired samples. METHODS.md defines averaging and NRMSD.

CITE
Ge, X. (2026). grase-glider-compare (v1.2.0). Zenodo.
https://doi.org/10.5281/zenodo.23194558

License: MIT.
