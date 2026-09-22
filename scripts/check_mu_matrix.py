"""Structurally verify a binned-mu release matrix, optionally against the legacy Union3 file.

Read-only. Checks the packing conventions that a consumer relies on, and compares the bin
redshifts against a reference file so that a drifted supernova selection shows up as a reported
difference rather than a silently different product.

Usage:
    uv run python scripts/check_mu_matrix.py output/<run>/mu_mat.fits
    uv run python scripts/check_mu_matrix.py output/<run>/mu_mat.fits \
        --reference other_cosmology/mu_mat_union3_cosmo=2.fits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def unpack(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path) as hdul:
        if len(hdul) != 1:
            raise SystemExit(f"{path}: expected a single HDU, found {len(hdul)}")
        data = np.asarray(hdul[0].data)
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise SystemExit(f"{path}: expected a square image, got shape {data.shape}")
    return data[0, 1:], data[1:, 0], data[1:, 1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--reference", type=Path, default=None, help="Legacy file to compare bin redshifts against.")
    args = ap.parse_args()

    with fits.open(args.path) as hdul:
        dtype, shape = hdul[0].data.dtype, hdul[0].data.shape
        corner = float(hdul[0].data[0, 0])
    zbins, mu, inv_cov = unpack(args.path)
    n = zbins.size

    checks: list[tuple[str, bool, str]] = []
    # FITS is big-endian on disk, so the dtype reads back as '>f8' rather than native float64.
    # Compare kind and width instead of the dtype object.
    is_f8 = dtype.kind == "f" and dtype.itemsize == 8
    checks.append(("single HDU, square, 64-bit float", is_f8, f"dtype {dtype}, shape {shape}"))
    checks.append(("corner [0,0] is zero", corner == 0.0, f"{corner!r}"))
    checks.append(("bin redshifts strictly increasing", bool(np.all(np.diff(zbins) > 0)), f"{n} bins"))
    checks.append(("inverse covariance symmetric", np.allclose(inv_cov, inv_cov.T, rtol=1e-10, atol=0), ""))

    eig = np.linalg.eigvalsh(inv_cov)
    checks.append(("inverse covariance positive definite", bool(eig.min() > 0), f"min eigenvalue {eig.min():.3g}"))

    cov = np.linalg.inv(inv_cov)
    sigma = np.sqrt(np.diag(cov))
    sane = bool(np.all((sigma > 0.001) & (sigma < 1.0)))
    checks.append(("per-bin sigma in 0.001-1 mag", sane, f"range {sigma.min():.4f}-{sigma.max():.4f} mag"))

    resid = bool(np.abs(mu).max() < 10.0)
    checks.append(("distance moduli look like residuals", resid, f"max |mu| = {np.abs(mu).max():.4f}"))

    ok = True
    for label, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{f'   ({detail})' if detail else ''}")

    if args.reference is not None:
        ref_z, _, _ = unpack(args.reference)
        print(f"\nAgainst {args.reference}:")
        if ref_z.size != n:
            print(f"  [DIFF] bin count {n} vs reference {ref_z.size} -- the SN selection differs.")
        else:
            worst = float(np.abs(ref_z - zbins).max())
            verdict = "identical" if worst < 1e-6 else f"differ by up to {worst:.4g}"
            print(f"  [INFO] bin redshifts {verdict}")
            if worst >= 1e-6:
                print("         A difference here means the filtered supernova set changed, not that")
                print("         the product is malformed. Investigate the selection before releasing.")

    print("\nRESULT:", "all structural checks passed" if ok else "STRUCTURAL CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
