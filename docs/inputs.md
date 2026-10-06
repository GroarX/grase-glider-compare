# Input guide

Copy an example JSON configuration and edit the `models`, `missions` and optional `groups` entries.

## Model data

Map `time`, `depth`, `lat`, `lon`, `temp` and `salt` under `variables`. Each file must contain both temperature and salinity on the same four physical dimensions. Multiple files are concatenated in time; merge separately delivered temperature/salinity files first. Remove extra ensemble dimensions and duplicate times.

Use a separable latitude/longitude grid and one-dimensional physical depth in metres. Convert native curvilinear or sigma-coordinate output before input. Subset global products to the mission region and dates, retaining cells and times that bracket observations. Missing cells and out-of-domain observations remain missing; the default maximum time gap is 48 hours.

Temperature must be °C. Set `temperature_kind` to `potential_0dbar` or `in_situ`; other temperature definitions require conversion. Salinity must be practical salinity. Confirm units and definitions from the exact input product.

## Forecast time format

Decoded valid UTC uses `model_pair.example.json` directly. For initialization plus lead time, use `forecast_pair.example.json` and map `init_coordinate`, `lead_coordinate` and `lead_units`. Select the same initialization for GrASE and NoGrASE and compare each cycle in a separate output folder.

## ESPC and GLORYS

Use `external_models.example.json` with `variant: "reference"` and IDs such as `ESPC_reference` or `GLORYS_reference`. The example maps ESPC `water_temp`/`salinity` and GLORYS `thetao`/`so`; check actual coordinate names and temperature definitions. Both templates assume potential temperature at 0 dbar.

Reference products receive their own comparison panels. Paired GrASE/NoGrASE error histograms are produced only when both variants are supplied. Select reference-product dates that overlap the glider observations.

## Glider data and groups

All supplied comparison results use `*_original_resolution.nc` QC-processed observations. The package recomputes their six-hour averages on each model's depth bins; it does not use the existing `*_6hrstats.nc` means for these results. [Shared input data](https://drive.google.com/drive/folders/1MJ9wrgEqYmXIEWtJ25PRUQ3-RaIrC-T9?usp=drive_link).

Original-resolution inputs require sample-aligned `time`, `depth`, `temperature`, `salinity`, `lat` and `lon`; alternate names can be mapped under `variables`. Temperature is in-situ °C and depth is positive downward.

For six-hour inputs, `stats_policy: "raw_companion"` recomputes exact averages from the original-resolution companion. `center_mean` reads the six-hour means directly but rebins them approximately with equal weights; original-resolution sections are then unavailable. Confirm the intended QC in the chosen input. The package preserves missing values and available QC flags; it does not add an automatic despiking pass.

List mission IDs in `groups`, for example `"yucatan_side": ["mission_a", "mission_b"]`. Groups pool paired time–depth cells, rather than averaging mission means.
