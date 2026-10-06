"""Publication-size 300-dpi figures; all continuous colormaps are cmocean."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cmocean.cm as cmo
import gsw
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap, Normalize

from .core import atomic_netcdf, metrics, moments
from .inputs import observation_paths, read_original

LOGGER = logging.getLogger(__name__)
WIDTH = 190 / 25.4
VARIABLES = ("temp", "salt", "sigma0")
LABELS = {
    "temp": "Temperature (°C)",
    "salt": "Practical salinity",
    "sigma0": r"$\sigma_0$ (kg m$^{-3}$)",
}
DIFF_LABELS = {
    "temp": r"$\Delta T$ (°C)",
    "salt": r"$\Delta S_P$",
    "sigma0": r"$\Delta\sigma_0$ (kg m$^{-3}$)",
}
CMAPS = {"temp": cmo.thermal, "salt": cmo.haline, "sigma0": cmo.dense}
LIMITS = {"temp": (4.0, 32.0), "salt": (34.5, 37.5), "sigma0": (20.0, 28.5)}
DLIMIT = {"temp": 5.0, "salt": 1.0, "sigma0": 1.5}
COLORS = [cmo.phase(v) for v in (0.08, 0.32, 0.57, 0.80)]
VARIANT_LABEL = {
    "grace": "GrASE",
    "nograce": "NoGrASE",
    "reference": "Reference products",
}


def style() -> None:
    """Consistent double-column sizes and legible 7–8 point sans-serif type."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1,
            "savefig.dpi": 300,
            "figure.dpi": 100,
            "pdf.fonttype": 42,
        }
    )


def save(
    fig: plt.Figure, path: Path, caption: str, registry: list[dict[str, Any]]
) -> None:
    """Write PNG and a companion caption, recording physical export dimensions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # The invisible full-figure rectangle makes tight export retain the chosen
    # physical width while retaining the required bbox_inches='tight' call.
    from matplotlib.patches import Rectangle

    boundary = Rectangle(
        (0, 0),
        1,
        1,
        transform=fig.transFigure,
        fill=False,
        edgecolor="none",
        linewidth=0,
    )
    fig.add_artist(boundary)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    path.with_suffix(".caption.txt").write_text(caption + "\n")
    registry.append(
        {"file": str(path), "dpi": 300, "width_mm": 190, "caption": caption}
    )
    LOGGER.info("Figure %s", path.name)


def panel(ax: plt.Axes, label: str, title: str = "") -> None:
    """Panel label and short title, avoiding titles in the plotting data area."""
    ax.set_title(f"({label}) {title}", loc="left", pad=4)
    ax.grid(True, lw=0.3, color="0.6", alpha=0.25)


def paired(ds: xr.Dataset, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Return matched observation/model arrays using the common T/S support."""
    mask = ds.paired_valid.values.astype(bool)
    return np.where(mask, ds["g" + var].values, np.nan), np.where(
        mask, ds[var].values, np.nan
    )


def concatenate(items: list[xr.Dataset]) -> xr.Dataset:
    """Pool actual time-depth pairs; retain duplicate UTC across different missions."""
    reference = items[0].depth.values
    if any(not np.array_equal(x.depth.values, reference) for x in items):
        raise ValueError(
            "Cannot pool unlike depth grids; explicitly rebin observations first"
        )
    return xr.concat(
        items,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).sortby("time")


def hov(
    ax: plt.Axes, ds: xr.Dataset, var: str, mode: str, time_labels: bool = True
) -> Any:
    """Time-depth bin colours retain actual support and temporal gaps.

    A grouped time-depth plot must be faceted by mission; the caller uses one
    mission at a time, never manufactures sequential/offset UTC times.
    """
    g, m = paired(ds, var)
    a = m - g if mode == "difference" else (g if mode == "glider" else m)
    times = mdates.date2num(ds.time.values)
    bounds = ds.depth_bounds.values
    edges = np.r_[bounds[:, 0], bounds[-1, 1]]
    cmap = cmo.balance if mode == "difference" else CMAPS[var]
    limits = (
        (-DLIMIT[var], DLIMIT[var]) if mode == "difference" else LIMITS[var]
    )
    # Each value represents its full physical depth bin and ±3-hour window.
    # Split at missing time windows rather than extending cells across gaps.
    cuts = np.r_[0, np.flatnonzero(np.diff(times) > 0.250001) + 1, len(times)]
    handle = None
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi <= lo:
            continue
        centres = times[lo:hi]
        xedges = np.r_[centres - 0.125, centres[-1] + 0.125]
        handle = ax.pcolormesh(
            xedges,
            edges,
            np.ma.masked_invalid(a[lo:hi].T),
            shading="flat",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )
    ax.set_ylim(1000, 0)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    date_format = "%b %d" if times[-1] - times[0] >= 3 else "%m/%d\n%H:%M"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    if time_labels:
        start_year = pd.Timestamp(ds.time.values[0]).year
        end_year = pd.Timestamp(ds.time.values[-1]).year
        years = (
            str(start_year)
            if start_year == end_year
            else f"{start_year}–{end_year}"
        )
        ax.set_xlabel(f"UTC ({years})")
    else:
        ax.tick_params(labelbottom=False)
    return handle


def ts_contours(ax: plt.Axes) -> None:
    """True SA–CT coordinates make sigma0 contours internally consistent."""
    sa, ct = np.meshgrid(np.linspace(34, 38, 100), np.linspace(2, 33, 100))
    contour = ax.contour(
        sa,
        ct,
        gsw.sigma0(sa, ct),
        levels=np.arange(20, 30),
        colors="0.68",
        linewidths=0.45,
    )
    ax.clabel(contour, fontsize=5, fmt="%.0f")
    ax.set(
        xlabel=r"Absolute salinity (g kg$^{-1}$)",
        ylabel="Conservative temperature (°C)",
        xlim=(34.5, 37.8),
        ylim=(3, 32),
    )


def ts_arrays(ds: xr.Dataset, model: bool) -> tuple[np.ndarray, np.ndarray]:
    """Convert paired SP/in-situ T to SA/CT at actual centroid coordinates."""
    prefix = "" if model else "g"
    sp, t = ds[prefix + "salt"].values, ds[prefix + "temp"].values
    p = gsw.p_from_z(-ds.depth.values[None, :], ds.lat_sample.values)
    sa = gsw.SA_from_SP(sp, p, ds.lon_sample.values, ds.lat_sample.values)
    ct = gsw.CT_from_t(sa, t, p)
    mask = ds.paired_valid.values.astype(bool)
    return np.where(mask, sa, np.nan), np.where(mask, ct, np.nan)


def ts_curve(
    ax: plt.Axes,
    ds: xr.Dataset,
    model: bool,
    color: Any,
    label: str,
    zrange: tuple[float, float] | None = None,
    scatter: bool = True,
) -> None:
    """T–S scatter and mean SA ± one std in 0.5 °C CT bins."""
    sa, ct = ts_arrays(ds, model)
    if zrange is not None:
        keep = (ds.depth.values >= zrange[0]) & (ds.depth.values < zrange[1])
        sa, ct = sa[:, keep], ct[:, keep]
    sa, ct = sa.ravel(), ct.ravel()
    valid = np.isfinite(sa) & np.isfinite(ct)
    sa, ct = sa[valid], ct[valid]
    if scatter and sa.size:
        step = max(1, int(np.ceil(sa.size / 2500)))
        ax.scatter(
            sa[::step],
            ct[::step],
            s=0.8,
            c=[color],
            alpha=0.12,
            rasterized=True,
        )
    centres = np.arange(3.25, 32, 0.5)
    mean, std = np.full(centres.size, np.nan), np.full(centres.size, np.nan)
    for i, c in enumerate(centres):
        a = sa[(ct >= c - 0.25) & (ct < c + 0.25)]
        if a.size >= 3:
            mean[i], std[i] = a.mean(), a.std(ddof=0)
    ax.plot(mean, centres, color=color, label=label, lw=0.9)
    ax.fill_betweenx(
        centres, mean - std, mean + std, color=color, alpha=0.12, linewidth=0
    )


def profile(
    ax: plt.Axes, ds: xr.Dataset, var: str, model: bool, color: Any, label: str
) -> None:
    """Mean ± one temporal/sample population std at native depth centres."""
    g, m = paired(ds, var)
    mean, std, n = moments(m if model else g)
    mean[n < 3], std[n < 3] = np.nan, np.nan
    z = ds.depth.values
    ax.plot(mean, z, color=color, label=label, ls="-", lw=1)
    ax.fill_betweenx(
        z, mean - std, mean + std, color=color, alpha=0.12, linewidth=0
    )
    ax.set(xlabel=LABELS[var], ylabel="Depth (m)", ylim=(1000, 0))


def plot_profiles(
    datasets: dict[str, xr.Dataset],
    title: str,
    path: Path,
    registry: list[dict[str, Any]],
) -> None:
    """One row per model, preserving each model's own paired observation support."""
    fig, axes = plt.subplots(
        len(datasets),
        4,
        figsize=(WIDTH, max(3.1, 2.2 * len(datasets))),
        layout="constrained",
    )
    axes = np.asarray(axes).reshape(len(datasets), 4)
    for i, (family, ds) in enumerate(datasets.items()):
        ts_contours(axes[i, 0])
        ts_curve(axes[i, 0], ds, False, "0.2", "Glider")
        ts_curve(axes[i, 0], ds, True, COLORS[i % 4], family)
        axes[i, 0].legend(loc="lower right", fontsize=5.5)
        for j, var in enumerate(VARIABLES, 1):
            profile(axes[i, j], ds, var, False, "0.2", "Glider")
            profile(axes[i, j], ds, var, True, COLORS[i % 4], family)
            if j > 1:
                axes[i, j].set_ylabel("")
        for j in range(4):
            panel(axes[i, j], chr(97 + i * 4 + j), family if j == 0 else "")
    first = next(iter(datasets.values()))
    if first.attrs.get("experiment") == "forecast":
        title += " | init " + first.attrs["forecast_initialization_utc"][
            :16
        ].replace("T", " ")
    fig.suptitle(title + " | mean ± 1 Std", fontsize=9)
    save(
        fig,
        path,
        "Paired T–S distributions and temperature, practical-salinity and potential-density-anomaly profiles. "
        "Rows correspond to the labelled models. Solid lines are means and shading is mean ± 1 standard deviation (ddof=0). "
        "Profiles pool all finite time–depth pairs within the selected missions, with at least three pairs per depth; "
        "missions receive weights proportional to their numbers of valid time bins. Each row uses that model's paired observation support. "
        "The T–S diagrams use Absolute Salinity and Conservative Temperature with sigma0 contours; curves and shading are mean SA ± 1 standard deviation "
        "in 0.5 °C CT bins. Points are deterministically thinned for display only. Density profiles use sigma0 (reference pressure 0 dbar).",
        registry,
    )


def plot_four_model(
    datasets: dict[str, xr.Dataset],
    title: str,
    path: Path,
    mode: str,
    registry: list[dict[str, Any]],
) -> None:
    """Four-system replacement for three_model_compare/state per mission."""
    nrow = len(datasets)
    fig, axes = plt.subplots(
        nrow,
        3,
        figsize=(WIDTH, max(2.8, 1.85 * nrow)),
        layout="constrained",
        squeeze=False,
    )
    for i, (family, ds) in enumerate(datasets.items()):
        for j, var in enumerate(VARIABLES):
            handle = hov(axes[i, j], ds, var, mode, i == nrow - 1)
            panel(axes[i, j], chr(97 + i * 3 + j), family)
            if j == 0:
                axes[i, j].set_ylabel("Depth (m)")
            else:
                axes[i, j].tick_params(labelleft=False)
            if i == nrow - 1:
                fig.colorbar(
                    handle,
                    ax=axes[:, j].tolist(),
                    orientation="horizontal",
                    fraction=0.025,
                    pad=0.045,
                    label=(
                        DIFF_LABELS[var]
                        if mode == "difference"
                        else LABELS[var]
                    ),
                    extend="both",
                )
    first = next(iter(datasets.values()))
    if first.attrs.get("experiment") == "forecast":
        title += (
            " | init "
            + first.attrs["forecast_initialization_utc"][:16].replace("T", " ")
            + " UTC"
        )
    fig.suptitle(
        title
        + (" | model − glider" if mode == "difference" else " | model state"),
        fontsize=9,
    )
    save(
        fig,
        path,
        "Time–depth comparison at glider sampling locations. Rows identify model experiments; columns show temperature, practical salinity and "
        "potential-density anomaly sigma0. "
        + (
            "Differences are model minus glider. "
            if mode == "difference"
            else "Shading represents model state on paired finite observation support. "
        )
        + "Time is UTC and depth increases downward. Colour cells use actual depth-bin edges and centred 6-hour windows; missing cells and temporal gaps are unfilled. "
        "Colour limits are common to all models and missions; colour-bar extensions indicate clipped display extremes. "
        "Input density is recomputed with TEOS-10; sigma0 is not in-situ density. Paired support can differ between models.",
        registry,
    )


def plot_assembled(
    items: dict[str, xr.Dataset],
    title: str,
    path: Path,
    registry: list[dict[str, Any]],
) -> None:
    """Mission-faceted model overview; concurrent UTC records never overlaid."""
    fig, axes = plt.subplots(
        len(items),
        3,
        figsize=(WIDTH, max(4, min(9, 1.4 * len(items)))),
        layout="constrained",
        squeeze=False,
    )
    for i, (mission, ds) in enumerate(items.items()):
        for j, var in enumerate(VARIABLES):
            handle = hov(axes[i, j], ds, var, "difference")
            panel(axes[i, j], str(i * 3 + j + 1), mission if j == 0 else "")
            if j == 0:
                axes[i, j].set_ylabel("Depth (m)")
            if i == len(items) - 1:
                fig.colorbar(
                    handle,
                    ax=axes[:, j].tolist(),
                    orientation="horizontal",
                    fraction=0.02,
                    pad=0.02,
                    label=DIFF_LABELS[var],
                    extend="both",
                )
    fig.suptitle(title, fontsize=9)
    save(
        fig,
        path,
        "Assembled model-minus-glider differences, faceted by mission to retain true UTC and prevent "
        "overplotting concurrent missions. Columns show temperature, practical salinity and sigma0 differences. "
        "Missing cells are unfilled. Colours use identical variable-specific limits in all panels. Each coloured cell is a paired "
        "6-hour-window/model-depth cell; this is not a count of independent physical casts.",
        registry,
    )


def plot_layer_ts(
    datasets: dict[str, xr.Dataset],
    title: str,
    path: Path,
    registry: list[dict[str, Any]],
) -> None:
    """25 depth layers × configured models, one panel per 40-m layer."""
    fig, axes = plt.subplots(5, 5, figsize=(WIDTH, 8.7), layout="constrained")
    for k, ax in enumerate(axes.flat):
        bounds = (k * 40.0, (k + 1) * 40.0)
        ts_contours(ax)
        for i, (family, ds) in enumerate(datasets.items()):
            # All model-specific observed means are visible in neutral shades;
            # they are not falsely represented as identical paired supports.
            ts_curve(
                ax,
                ds,
                False,
                str(0.15 + 0.12 * i),
                "Glider / " + family,
                bounds,
                False,
            )
            ts_curve(ax, ds, True, COLORS[i % 4], family, bounds, True)
        panel(ax, str(k + 1), f"{int(bounds[0])}–{int(bounds[1])} m")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=5.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=5.5,
        bbox_to_anchor=(0.5, -0.055),
    )
    fig.supxlabel(r"Absolute salinity (g kg$^{-1}$)", fontsize=8)
    fig.supylabel("Conservative temperature (°C)", fontsize=8)
    fig.suptitle(title + " | 40-m depth layers", fontsize=9)
    save(
        fig,
        path,
        "T–S comparison in 25 consecutive 40-m layers from 0 to 1,000 m. "
        "Each panel includes the configured models and their paired glider distributions. Colours identify model families; "
        "neutral curves show each model's corresponding glider support. Curves and bands are mean Absolute Salinity ± 1 standard deviation "
        "in 0.5 °C Conservative Temperature bins (minimum three samples). Grey contours are TEOS-10 sigma0. "
        "Layers without enough data remain empty. Depth membership uses native model bin centres; observations below 10 m were selected.",
        registry,
    )


def plot_hist_depth(
    nograse: xr.Dataset,
    grase: xr.Dataset,
    title: str,
    path: Path,
    registry: list[dict[str, Any]],
) -> None:
    """Paired GrASE/NoGrASE difference histograms with 50-m layer colours."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.8), layout="constrained")

    # Align on (mission_id, UTC) explicitly, avoiding duplicate UTC ambiguity.
    def keys(ds: xr.Dataset) -> list[str]:
        return [
            str(m) + "|" + str(t)
            for m, t in zip(ds.mission_id.values, ds.time.values)
        ]

    kn, kg = keys(nograse), keys(grase)
    lookup = {v: i for i, v in enumerate(kg)}
    ni = [i for i, v in enumerate(kn) if v in lookup]
    gi = [lookup[kn[i]] for i in ni]
    nn, gg = nograse.isel(time=ni), grase.isel(time=gi)
    if not np.array_equal(nn.depth.values, gg.depth.values):
        raise ValueError(
            "Paired variant histogram requires the same physical depth centres"
        )
    support = nn.paired_valid.values.astype(
        bool
    ) & gg.paired_valid.values.astype(bool)
    cmap, norm = ListedColormap(
        cmo.deep(np.linspace(0.18, 1, 256))
    ), Normalize(0, 1000)
    for j, var in enumerate(VARIABLES):
        allvals = np.concatenate(
            [(d[var].values - d["g" + var].values)[support] for d in (nn, gg)]
        )
        if allvals.size:
            bound = max(DLIMIT[var], float(np.max(np.abs(allvals))))
        else:
            bound = DLIMIT[var]
        bins = np.linspace(-bound, bound, 61)
        for i, ds in enumerate((nn, gg)):
            diff = np.where(
                support, ds[var].values - ds["g" + var].values, np.nan
            )
            ax = axes[i, j]
            for z in range(0, 1000, 50):
                values = diff[
                    :, (ds.depth.values >= z) & (ds.depth.values < z + 50)
                ].ravel()
                values = values[np.isfinite(values)]
                if values.size:
                    ax.hist(
                        values,
                        bins=bins,
                        histtype="step",
                        linewidth=0.65,
                        color=cmap(norm(z + 25)),
                    )
            ax.axvline(0, color="0.4", lw=0.6, ls="--")
            ax.set(xlabel=DIFF_LABELS[var], ylabel="Count" if j == 0 else "")
            panel(
                ax,
                chr(97 + i * 3 + j),
                ("NoGrASE" if i == 0 else "GrASE") + f" (n={support.sum():,})",
            )
    for j in range(3):
        ymax = max(axes[0, j].get_ylim()[1], axes[1, j].get_ylim()[1])
        axes[0, j].set_ylim(0, ymax)
        axes[1, j].set_ylim(0, ymax)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.04,
        label="50-m depth-layer centre (m)",
    )
    fig.suptitle(title, fontsize=9)
    save(
        fig,
        path,
        "Histograms of model-minus-glider temperature, practical-salinity and sigma0 differences. "
        "Rows compare NoGrASE and GrASE on the joint finite support shared by both variants at identical mission, UTC and depth. "
        "Colours identify 50-m depth layers. The same 60 histogram intervals are used for both variants of each variable and cover all finite values. "
        "Counts represent time–depth cells, not statistically independent casts. Each panel gives the total common-support sample count.",
        registry,
    )


def plot_state_hist(
    datasets: dict[str, xr.Dataset],
    title: str,
    path: Path,
    registry: list[dict[str, Any]],
) -> None:
    """Compact distribution comparison, complementing difference histograms."""
    fig, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(WIDTH, max(2.8, 1.75 * len(datasets))),
        layout="constrained",
        squeeze=False,
    )
    for i, (family, ds) in enumerate(datasets.items()):
        for j, var in enumerate(VARIABLES):
            g, m = paired(ds, var)
            g, m = g[np.isfinite(g)], m[np.isfinite(m)]
            bins = np.linspace(
                min(g.min(), m.min()), max(g.max(), m.max()) + 1e-10, 51
            )
            for a, color, label in (
                (g, "0.2", "Glider"),
                (m, COLORS[i % 4], family),
            ):
                axes[i, j].hist(
                    a,
                    bins=bins,
                    density=True,
                    histtype="step",
                    color=color,
                    label=label,
                )
            axes[i, j].set(
                xlabel=LABELS[var],
                ylabel="Probability density" if j == 0 else "",
            )
            panel(axes[i, j], chr(97 + i * 3 + j), family + f" (n={g.size:,})")
            if j == 0:
                axes[i, j].legend(fontsize=5.5)
    fig.suptitle(title, fontsize=9)
    save(
        fig,
        path,
        "Distributions of paired glider and model temperature, practical salinity and sigma0. "
        "Each row uses the labelled model's own joint finite T/S support and identical histogram intervals for model and glider. "
        "Histograms integrate to one; y-axis units are the inverse of the corresponding x-axis units. Counts are valid time–depth pairs. "
        "Pooling weights missions and depths by available pairs and does not provide volume-weighted water-mass fractions.",
        registry,
    )


def plot_nrmsd(
    records: dict[str, xr.Dataset],
    title: str,
    path: Path,
    registry: list[dict[str, Any]],
    normalization: dict[str, list[float]],
) -> None:
    """Legacy min–max NRMSD, obs-std NRMSD and count; no curve editing."""
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 7), layout="constrained")
    colors = cmo.phase(np.linspace(0.05, 0.9, len(records)))
    for color, (label, ds) in zip(colors, records.items()):
        for j, var in enumerate(VARIABLES):
            stats = metrics(*paired(ds, var))
            lo, hi = normalization[var]
            scaled = (
                (stats["rmsd"] - lo) / (hi - lo)
                if hi > lo
                else np.full(ds.sizes["depth"], np.nan)
            )
            for i, a in enumerate(
                (scaled, stats["nrmsd_obs_std"], stats["n"])
            ):
                axes[i, j].plot(
                    a, ds.depth.values, color=color, lw=0.8, label=label
                )
                axes[i, j].set_ylim(1000, 0)
                if i == 0:
                    axes[i, j].set_xlim(0, 1)
                axes[i, j].set_xlabel(
                    (
                        "NRMSD (global min–max)",
                        "RMSD / observed std",
                        "Paired count",
                    )[i]
                )
                if j == 0:
                    axes[i, j].set_ylabel("Depth (m)")
                panel(axes[i, j], chr(97 + i * 3 + j), LABELS[var])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=5.5,
    )
    fig.suptitle(title, fontsize=9)
    save(
        fig,
        path,
        "Vertical errors for temperature, practical salinity and sigma0. Top: legacy NRMSD=(RMSD−RMSD_min)/(RMSD_max−RMSD_min), "
        "where each variable's extrema are fixed across the selected full set of individual mission/model caches. "
        "Middle: RMSD divided by the paired glider standard deviation at each depth; undefined for zero variance. "
        "Bottom: finite paired counts. RMSD requires at least three pairs. No time interpolation of missing observation bins, "
        "depth despiking, clipping or replacement of error curves is applied. Native depth grids are retained. "
        "Normalization bounds are recorded in the metrics manifest.",
        registry,
    )


def plot_raw(
    raw: xr.Dataset, title: str, path: Path, registry: list[dict[str, Any]]
) -> None:
    """Original-resolution QC observation time–depth panels; display thinning only."""
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.4), layout="constrained")
    for i, var in enumerate(VARIABLES):
        values = raw[var].values
        valid = np.flatnonzero(np.isfinite(values))
        step = max(1, int(np.ceil(valid.size / 120000)))
        idx = valid[::step]
        handle = axes[i].scatter(
            raw.time.values[idx],
            raw.depth.values[idx],
            c=values[idx],
            s=0.6,
            cmap=CMAPS[var],
            vmin=LIMITS[var][0],
            vmax=LIMITS[var][1],
            linewidths=0,
        )
        axes[i].set(ylim=(1000, 0), ylabel="Depth (m)")
        loc = mdates.AutoDateLocator(minticks=4, maxticks=6)
        axes[i].xaxis.set_major_locator(loc)
        axes[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        panel(axes[i], chr(97 + i), LABELS[var])
        fig.colorbar(
            handle, ax=axes[i], label=LABELS[var], extend="both", pad=0.015
        )
    axes[-1].set_xlabel("UTC")
    fig.suptitle(title + " | QC observations, original resolution", fontsize=9)
    save(
        fig,
        path,
        "Original-resolution QC glider temperature, practical salinity and TEOS-10 sigma0 versus UTC and depth. "
        "Points are deterministically thinned to at most 120,000 per variable for display; all eligible source samples are used for bin averaging. "
        "Missing observations are not interpolated. Sigma0 is computed from each sample's T/S/depth/position and is not the source in-situ density variable.",
        registry,
    )


def render(
    config: dict[str, Any],
    output: Path,
    models: list[str] | None,
    missions: list[str] | None,
    groups: list[str] | None,
    skip_individual: bool,
    config_root: Path | None = None,
) -> None:
    """Render nonredundant families and write exact numerical metrics to NetCDF."""
    style()
    np.random.seed(config["settings"]["seed"])
    specs = [s for s in config["models"] if not models or s["id"] in models]
    mission_ids = [
        s["id"]
        for s in config["missions"]
        if not missions or s["id"] in missions
    ]
    loaded: dict[tuple[str, str], xr.Dataset] = {}
    for model in specs:
        for mission in mission_ids:
            path = output / "cache" / f"{model['id']}_{mission}_bins.nc"
            with xr.open_dataset(path) as ds:
                loaded[(model["id"], mission)] = ds.load()
    registry: list[dict[str, Any]] = []
    figures = output / "figures"
    if config_root is not None and not skip_individual:
        for mission in config["missions"]:
            if mission["id"] not in mission_ids:
                continue
            paths, kind = observation_paths(
                mission, config_root, config["settings"]["stats_policy"]
            )
            if kind == "original":
                raw = read_original(paths, mission, config["settings"])
                plot_raw(
                    raw,
                    mission["id"],
                    figures
                    / mission["id"]
                    / f"glider_original_{mission['id']}.png",
                    registry,
                )
                del raw
    scopes = {} if skip_individual else {m: [m] for m in mission_ids}
    for name, ids in config.get("groups", {}).items():
        if (not groups or name in groups) and set(ids) <= set(mission_ids):
            scopes[name] = ids
    normalization = {}
    for var in VARIABLES:
        arrays = [metrics(*paired(ds, var))["rmsd"] for ds in loaded.values()]
        vals = np.concatenate(arrays)
        vals = vals[np.isfinite(vals)]
        normalization[var] = [float(vals.min()), float(vals.max())]
    metric_manifest: dict[str, Any] = {
        "normalization": normalization,
        "records": [],
    }
    # Store metrics for every scope including groups using pooled pairs, never averages of mission means.
    for scope, ids in scopes.items():
        pooled = {
            m["id"]: concatenate(
                [loaded[(m["id"], mission)] for mission in ids]
            )
            for m in specs
        }
        for mid, ds in pooled.items():
            data = {}
            for var in VARIABLES:
                for name, a in metrics(*paired(ds, var)).items():
                    data[var + "_" + name] = ("depth", a)
            metrics_ds = xr.Dataset(
                data,
                coords={"depth": ds.depth.values},
                attrs={
                    "scope": scope,
                    "model": mid,
                    "pooling": "Equal weight per paired time-depth cell",
                    "normalization_bounds": json.dumps(normalization),
                },
            )
            metric_path = output / "metrics" / f"{mid}_{scope}.nc"
            # Numerical metrics are deterministically refreshed only within a new run directory.
            if not metric_path.exists():
                atomic_netcdf(metrics_ds, metric_path)
            metric_manifest["records"].append(
                {
                    "model": mid,
                    "scope": scope,
                    "pairs": int(ds.paired_valid.sum()),
                    "metrics": str(metric_path),
                }
            )
        for variant in ("nograce", "grace", "reference"):
            ds4 = {
                m["family"]: pooled[m["id"]]
                for m in specs
                if m["variant"] == variant
            }
            if not ds4:
                continue
            title = f"{scope} | {VARIANT_LABEL[variant]}"
            prefix = "four_model" if len(ds4) == 4 else f"{len(ds4)}_model"
            plot_profiles(
                ds4,
                title,
                figures
                / scope
                / f"{prefix}_TS_profiles_{variant}_{scope}.png",
                registry,
            )
            plot_layer_ts(
                ds4,
                title,
                figures / scope / f"{prefix}_TS25panels_{variant}_{scope}.png",
                registry,
            )
            plot_state_hist(
                ds4,
                title,
                figures
                / scope
                / f"{prefix}_state_histograms_{variant}_{scope}.png",
                registry,
            )
            if len(ids) == 1:
                for mode, tag in (
                    ("difference", "compare"),
                    ("model", "state"),
                ):
                    plot_four_model(
                        ds4,
                        title,
                        figures
                        / scope
                        / f"{prefix}_{tag}_{variant}_{scope}.png",
                        mode,
                        registry,
                    )
            else:
                for m in specs:
                    if m["variant"] == variant:
                        parts = {
                            mission: loaded[(m["id"], mission)]
                            for mission in ids
                        }
                        plot_assembled(
                            parts,
                            f"{m['id']} | {scope}",
                            figures / scope / f"{m['id']}_{scope}_bins.png",
                            registry,
                        )
        families = list(dict.fromkeys(m["family"] for m in specs))
        for family in families:
            no, yes = family + "_nograce", family + "_grace"
            if no in pooled and yes in pooled:
                plot_hist_depth(
                    pooled[no],
                    pooled[yes],
                    f"{family} | {scope}",
                    figures
                    / scope
                    / f"{family}_{scope}_diff_hist_depth50m.png",
                    registry,
                )
        if len(ids) > 1:
            plot_nrmsd(
                pooled,
                scope,
                figures
                / scope
                / f"nrmsd_vs_depth_hindcasts_grouped_{scope}.png",
                registry,
                normalization,
            )
    if not skip_individual:
        for family in dict.fromkeys(s["family"] for s in specs):
            for variant in ("nograce", "grace", "reference"):
                mid = family + "_" + variant
                if mid not in {s["id"] for s in specs}:
                    continue
                records = {m: loaded[(mid, m)] for m in mission_ids}
                plot_nrmsd(
                    records,
                    mid,
                    figures
                    / "nrmsd"
                    / f"nrmsd_vs_depth_hindcasts_individual_paper_{mid}.png",
                    registry,
                    normalization,
                )
    (output / "figure_manifest.json").write_text(
        json.dumps(registry, indent=2) + "\n"
    )
    (output / "metrics_manifest.json").write_text(
        json.dumps(metric_manifest, indent=2) + "\n"
    )
    (output / "configuration.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    LOGGER.info("Rendered %d figures", len(registry))
