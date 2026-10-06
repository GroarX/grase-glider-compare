"""Scientific definitions; metres positive down, UTC, in-situ degrees Celsius."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import gsw
import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def depth_edges(depth: np.ndarray) -> np.ndarray:
    """Legacy midpoint bins: outer edges equal endpoint centres, [lo, hi)."""
    z = np.asarray(depth, dtype=float)
    if z.size < 2 or not np.all(np.diff(z) > 0):
        raise ValueError("Need at least two strictly increasing depth centres")
    return np.r_[z[0], (z[:-1] + z[1:]) / 2, z[-1]]


def density(
    sp: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """TEOS-10 potential-density anomaly sigma0 (kg m-3), not in-situ density."""
    p = gsw.p_from_z(-np.asarray(z), lat)
    sa = gsw.SA_from_SP(sp, p, lon, lat)
    return np.asarray(gsw.sigma0(sa, gsw.CT_from_t(sa, t, p)))


def insitu(
    pt: np.ndarray,
    sp: np.ndarray,
    z: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Potential temperature referenced to 0 dbar -> in-situ temperature."""
    p = gsw.p_from_z(-np.asarray(z), lat)
    sa = gsw.SA_from_SP(sp, p, lon, lat)
    return np.asarray(gsw.t_from_CT(sa, gsw.CT_from_pt(sa, pt), p))


def moments(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, ...]:
    """Finite population mean/std and number of samples, without gap filling."""
    a = np.asarray(values, dtype=float)
    valid = np.isfinite(a)
    n = valid.sum(axis=axis)
    mean = np.divide(
        np.where(valid, a, 0).sum(axis=axis),
        n,
        out=np.full(n.shape, np.nan),
        where=n > 0,
    )
    residual = a - np.expand_dims(mean, axis)
    var = np.divide(
        np.where(valid, residual**2, 0).sum(axis=axis),
        n,
        out=np.full(n.shape, np.nan),
        where=n > 0,
    )
    return mean, np.sqrt(var), n


def metrics(
    g: np.ndarray, m: np.ndarray, min_pairs: int = 3
) -> dict[str, np.ndarray]:
    """Paired bias/RMSD/r and RMSD/observation std at each depth (ddof=0)."""
    ok = np.isfinite(g) & np.isfinite(m)
    gg, mm = np.where(ok, g, np.nan), np.where(ok, m, np.nan)
    gm, gs, n = moments(gg)
    mmn, ms, _ = moments(mm)
    bias, _, _ = moments(mm - gg)
    mse, _, _ = moments((mm - gg) ** 2)
    covariance, _, _ = moments((gg - gm) * (mm - mmn))
    with np.errstate(divide="ignore", invalid="ignore"):
        rmsd = np.sqrt(mse)
        r = covariance / (gs * ms)
        nrmsd = rmsd / gs
    result = {"bias": bias, "rmsd": rmsd, "r": r, "nrmsd_obs_std": nrmsd}
    for name, a in result.items():
        result[name] = np.where((n >= min_pairs) & np.isfinite(a), a, np.nan)
    result["n"] = n
    return result


def signature(paths: list[Path], settings: dict[str, Any]) -> str:
    """Invalidate caches after input size/mtime, configuration or code changes."""
    sources = [
        (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
        for p in paths
    ]
    code = hashlib.sha256()
    for p in [
        Path(__file__).parent / name
        for name in ("core.py", "inputs.py", "collocate.py", "forecast.py")
    ]:
        code.update(
            ast.dump(
                ast.parse(p.read_text()), include_attributes=False
            ).encode()
        )
    payload = json.dumps([sources, settings, code.hexdigest()], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write a complete cache atomically, preserving an older differing cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.nc")
    encoding = {
        v: {"zlib": True, "complevel": 3}
        for v in ds.data_vars
        if ds[v].dtype.kind in "fi"
    }
    ds.to_netcdf(temp, engine="netcdf4", encoding=encoding)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {path}; use a new output root"
        )
    temp.rename(path)
    LOGGER.info("Saved %s", path)
