"""Binned distance-modulus data-release products.

`write_release_products` writes two files side by side.

`mu_mat.fits` is one (n_bins + 1) x (n_bins + 1) float64 image in the packed layout used by past
Union releases:

    whole[0, 0]   = 0
    whole[0, 1:]  = zbins                       bin redshifts
    whole[1:, 0]  = median(mu_zbins, axis=0)    binned distance modulus per bin
    whole[1:, 1:] = inv(cov(mu_zbins))          inverse covariance

Row and column zero are metadata, not part of the matrix. `mu_binned.ecsv` holds the same content
as a table, with the covariance rather than its inverse, plus the run provenance.

The distance moduli are residuals relative to FlatLambdaCDM(H0=70, Om0=0.3), the fiducial
subtracted when the bins are built (`unity.data.loaders._get_redshift_bins`). The bin mean is
pinned near zero by the z=0 anchor of that interpolation basis, so cosmology fits should carry a
free magnitude offset (scriptM). The data-release section of README.md covers both points.

The layout is byte-compatible with `other_cosmology/mu_mat_union3_cosmo=2.fits`, written by the
retired pystan script `scripts/read_and_sample.py` (lines 960-966) via an external
`DavidsNM.save_img` helper that is not part of this repository. Existing consumers keep working:
`other_cosmology/compute_chi2s.py` reads only `hdu.data`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from astropy.io import fits
from astropy.table import Table
from loguru import logger  # direct import: unity/__init__ imports main, which imports this module

#: Fiducial cosmology the binned distance moduli are residuals against.
FIDUCIAL_H0 = 70.0
FIDUCIAL_OM0 = 0.3

_INDEX_RE = re.compile(r"\[(\d+)\]$")


def extract_mu_zbins(samples: pl.DataFrame, n_bins: int) -> np.ndarray:
    """Pull the `mu_zbins[i]` columns out of saved draws as an (n_draws, n_bins) array.

    The columns are 1-based strings written by the NumPyro sample extraction. They must be ordered
    by the integer inside the bracket: sorting them as text puts `mu_zbins[10]` before
    `mu_zbins[2]` and silently scrambles the bins, which is the easiest way to ship a wrong
    covariance from a run that otherwise looks healthy.
    """
    indexed: list[tuple[int, str]] = []
    for name in samples.columns:
        if not name.startswith("mu_zbins["):
            continue
        match = _INDEX_RE.search(name)
        if match is None:
            raise ValueError(f"Unparseable mu_zbins column name: {name!r}")
        indexed.append((int(match.group(1)), name))

    if not indexed:
        raise ValueError(
            "No mu_zbins[...] columns in the saved draws. The run needs "
            "extra_vector_parameters_to_save: ['mu_zbins'] (and cosmology_model 'binned_mu')."
        )

    indexed.sort(key=lambda pair: pair[0])
    found = [i for i, _ in indexed]
    if found != list(range(1, n_bins + 1)):
        raise ValueError(
            f"Expected mu_zbins[1..{n_bins}] in the draws, found indices {found[:5]}...{found[-3:]} "
            f"({len(found)} columns). The chains and the redshift bins disagree."
        )
    return samples.select([name for _, name in indexed]).to_numpy()


def build_mu_matrix(mu_zbins: np.ndarray, zbins: np.ndarray) -> np.ndarray:
    """Pack draws (n_draws, n_bins) and bin redshifts into the legacy (n_bins+1)^2 matrix."""
    mu_zbins = np.asarray(mu_zbins, dtype=np.float64)
    zbins = np.asarray(zbins, dtype=np.float64)

    if mu_zbins.ndim != 2:
        raise ValueError(f"mu_zbins must be (n_draws, n_bins); got shape {mu_zbins.shape}.")
    n_draws, n_bins = mu_zbins.shape
    if n_bins != zbins.size:
        raise ValueError(f"mu_zbins has {n_bins} bins but zbins has {zbins.size}.")
    if n_draws <= 10 * n_bins:
        raise ValueError(
            f"Only {n_draws} draws for {n_bins} bins. The sample covariance is near-singular "
            f"below ~10 draws per bin; inverting it would produce a meaningless precision matrix."
        )

    mu_cov = np.cov(mu_zbins.T)
    condition = float(np.linalg.cond(mu_cov))
    logger.info(f"Binned mu covariance: {n_bins} bins from {n_draws} draws, condition number {condition:.3g}.")
    if condition > 1e12:
        logger.warning(
            f"Binned mu covariance is poorly conditioned ({condition:.3g}); the inverse stored in "
            f"the release matrix may be numerically unreliable."
        )

    whole = np.zeros((n_bins + 1, n_bins + 1), dtype=np.float64)
    whole[1:, 1:] = np.linalg.inv(mu_cov)
    whole[1:, 0] = np.median(mu_zbins, axis=0)
    whole[0, 1:] = zbins
    return whole


def _header(provenance: dict[str, Any] | None) -> fits.Header:
    header = fits.Header()
    for line in (
        "Union3 / UNITY 1.8 binned distance-modulus release matrix.",
        "Layout: [0,0]=0; [0,1:]=bin redshifts; [1:,0]=binned distance modulus;",
        "        [1:,1:]=INVERSE covariance, for backward compatibility with",
        "        past Union releases.",
        f"Distance moduli are RESIDUALS w.r.t. FlatLambdaCDM(H0={FIDUCIAL_H0:g}, Om0={FIDUCIAL_OM0:g});",
        "add that fiducial back to recover absolute distance moduli.",
        "mu_binned.ecsv ships the covariance.",
        "Fit with a free magnitude offset (scriptM). The bin mean is a",
        "normalisation, not a measurement of absolute scale.",
    ):
        header.add_comment(line)
    for key, value in (provenance or {}).items():
        header.add_comment(f"{key}: {value}")
    return header


def write_release_products(
    samples: pl.DataFrame,
    zbins: np.ndarray,
    output_dir: Path,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write mu_mat.fits and mu_binned.ecsv into `output_dir`; return the paths written."""
    zbins = np.asarray(zbins, dtype=np.float64)
    mu_zbins = extract_mu_zbins(samples, zbins.size)
    mu_cov = np.cov(mu_zbins.T)

    # The ECSV is written FIRST, deliberately. It needs no matrix inversion, so it cannot fail the
    # way the packed matrix can, and it is the only place the bin redshifts are persisted -- they
    # are derived from the filtered supernova set at load time and never saved with the chains. If
    # the inversion below fails, this file still lets the product be rebuilt without re-running.
    table = Table(
        {
            "z": zbins,
            "mu_residual": np.median(mu_zbins, axis=0),
            "mu_err": np.sqrt(np.diag(mu_cov)),
            "covariance": mu_cov,
        }
    )
    table.meta["description"] = (
        f"Union3/UNITY 1.8 binned distance moduli. mu_residual is relative to "
        f"FlatLambdaCDM(H0={FIDUCIAL_H0:g}, Om0={FIDUCIAL_OM0:g}); add that fiducial back for absolute "
        f"distance moduli. 'covariance' is the full {zbins.size}x{zbins.size} covariance; mu_mat.fits "
        f"ships its inverse for backward compatibility with past Union releases. "
        f"mu_err is the square root of its diagonal. Fit with a free magnitude offset (scriptM). "
        f"The bin mean is a normalisation, not a measurement of absolute scale."
    )
    for key, value in (provenance or {}).items():
        table.meta[key] = str(value)
    ecsv_path = output_dir / "mu_binned.ecsv"
    table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    logger.info(f"Wrote binned distance moduli and covariance to {ecsv_path}.")

    whole = build_mu_matrix(mu_zbins, zbins)
    fits_path = output_dir / "mu_mat.fits"
    fits.PrimaryHDU(data=whole, header=_header(provenance)).writeto(fits_path, overwrite=True)
    logger.info(f"Wrote {whole.shape[0]}x{whole.shape[1]} binned-mu release matrix to {fits_path}.")

    return {"mu_binned": ecsv_path, "mu_mat": fits_path}
