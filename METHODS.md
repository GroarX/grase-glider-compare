# Comparison methods

## Observation averaging and interpolation

The default depth range is 10–1,000 m. Observation windows are centred on 00, 06, 12 and 18 UTC and include samples in [centre − 3 hours, centre + 3 hours). Each bin requires at least two jointly finite temperature–salinity samples.

Depth bins use the model's positive depth centres from 1 to 1,000 m. Interior edges are adjacent-centre midpoints; the outer edges equal the first and last centres. Bins include their lower edge and exclude their upper edge. A bin centre can fall outside the observation depth range if part of its bin overlaps that range.

Glider temperature, practical salinity and σ₀ are averaged from raw samples. Observation σ₀ is calculated before averaging. The model is evaluated at the paired samples' mean UTC, longitude and latitude within each bin, at the corresponding native model depth. This is interpolation to the bin centroid, rather than averaging model values evaluated at every raw sample.

Horizontal interpolation is bilinear and time interpolation is linear. All nonzero-weight corners must be finite. There is no spatial or temporal extrapolation. The default maximum gap between bracketing model times is 48 hours, adjustable with `max_time_gap_hours`; exact model timestamps remain usable.

Existing missing values and temperature/salinity QC flags are retained; where QC flags are present, only flag 1 is used. Optional mission-specific `exclude_intervals` specify inclusive `start_utc`, `end_utc`, a `reason` and an optional `expected_samples` count. Exclusions are recorded in the cache and do not modify source files. Sample-level exclusions require original-resolution observations.

For `*_6hrstats.nc`, `stats_policy: "raw_companion"` rebins the matching original-resolution samples. The alternative `center_mean` policy equally weights existing source-bin means within each model bin. It cannot recover raw sample counts or within-bin variance; its density is calculated from remapped mean temperature and salinity.

## Variables and pairing

Model potential temperature referenced to 0 dbar is converted to in-situ temperature using TEOS-10. Temperature profiles and differences use in-situ °C; salinity profiles and differences use practical salinity. Density is potential-density anomaly σ₀, referenced to 0 dbar, in kg m⁻³. T–S diagrams use Absolute Salinity in g kg⁻¹ and Conservative Temperature in °C with σ₀ contours.

Each comparison requires finite temperature, salinity and σ₀ from both model and glider. Each model therefore has its own paired observation support. GrASE/NoGrASE difference histograms additionally require common mission, UTC and depth across both variants.

Mean profiles and errors require at least three pairs per depth. Shading represents ±1 population standard deviation (`ddof=0`), not a confidence interval. Multiple missions are pooled before statistics are calculated, with equal weight per paired time–depth cell. Mission IDs and original UTC are retained.

Forecasts use valid UTC, calculated from initialization plus lead time or read directly from the input. Both variants must use the same initialization. Separate forecast cycles are not combined. Pooling across a mission does not control for forecast lead time.

## Error metrics

For paired model values m and glider values g at a given depth:

```text
bias = mean(m − g)
RMSD = sqrt(mean((m − g)²))
NRMSD_range = (RMSD − RMSD_min) / (RMSD_max − RMSD_min)
NRMSD_std = RMSD / std(g)
```

For `NRMSD_range`, the minimum and maximum are taken separately for temperature, salinity and σ₀ across the selected individual model–mission–depth comparisons. This metric depends on which comparisons are selected. `NRMSD_std` uses the observed population standard deviation at each depth. Undefined normalizations and correlations remain missing. Counts and Pearson correlations are also saved.

Glider observations assimilated in GrASE are not independent validation data. Paired time–depth cells should not be interpreted as independent physical casts.

## Figures

All figures use 300 dpi, 190-mm width and cmocean colourmaps: thermal for temperature, haline for salinity, dense for σ₀, balance for differences and deep for depth. Profiles use solid mean lines and standard-deviation shading.

Sections use actual depth-bin boundaries and 6-hour window edges. Missing cells and gaps remain blank. Colourbar extensions indicate values beyond the displayed colour range. Histograms include all finite paired values. Points may be deterministically thinned for display, but statistics use all eligible samples.
