"""Stan vs NumPyro posterior comparison for unity_1.8, driven off the paired
configs src/unity/configs/unity_1_8_comparison_{stan,numpyro}.yml.

Both configs run the full published Union3.1+UNITY1.8 selection (2085 SNe,
no distance ladder -> H0 unconstrained -> safe to run unblinded), 4 parallel
chains x 5000 draws each (same chain count on both so wall-clock is directly
comparable). Unblind flags are runtime-only per policy -- never put them in a
config file -- so run the two samplers first, one at a time, timing each
(--extra_single_dimension_parameters_only false is required so the per-SN
latents are saved):

    time uv run unity --base unity_1_8_comparison_stan.yml --blinding none --really_unblind true --extra_single_dimension_parameters_only false
    time uv run unity --base unity_1_8_comparison_numpyro.yml --blinding none --really_unblind true --extra_single_dimension_parameters_only false

then:

    uv run python scripts/compare_stan_numpyro.py

If a chain lands in a different self-consistent fast/slow standardization mode
(the known label-switching issue -- shows up as huge R-hat on beta_R_low /
this_MB_slow / delta_0 etc., a regular UNITY occurrence on both samplers),
exclude it per side and rerun, e.g.:

    uv run python scripts/compare_stan_numpyro.py --np-chains 0,1,3 --stan-chains 0,1,2,3

The chain-health / per-parameter R-hat output is what tells you which chains
(if any) to drop; a comparison including a mode-trapped chain on one side
will fail the z-test for reasons unrelated to the port.

Phase 1 compares the scalar (bracket-free) parameters with the per-parameter
table below; phase 2 compares every shared per-SN latent column ("name[i]"),
reading both parquets in column chunks via pyarrow (each file is ~6+ GB; never
load them whole) and using a vectorized implementation of the same Geyer-ESS
z-test. Full per-latent results are written to
output/unity_1_8_comparison/latents_comparison.parquet.

Reuses the z-test/Bonferroni-calibrated verdict from
scripts/numpyro_port/run_comparison.py (MC error via per-chain Geyer ESS).
lp__ is NOT cross-compared: NumPyro's differs from Stan's by dropped
constants and by the two engines' different simplex-transform Jacobians;
parameter posteriors are invariant.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "numpyro_port"))
from run_comparison import ess_geyer, split_rhat  # noqa: E402

ROOT = HERE.parent
STAN_OUT = ROOT / "output" / "unity_1_8_comparison" / "stan" / "mcmc_samples.parquet"
NUMPYRO_OUT = ROOT / "output" / "unity_1_8_comparison" / "numpyro" / "mcmc_samples.parquet"
LATENTS_RESULT = ROOT / "output" / "unity_1_8_comparison" / "latents_comparison.parquet"
NUM_CHAINS = 4
NUM_DRAWS = 5000
SKIP = {"lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__",
        "divergent__", "energy__", "chain__", "iter__", "draw__"}
CHUNK = 256  # latent columns per read; bounds peak memory at a few hundred MB

# Chain subsets to compare (set from --np-chains/--stan-chains; None = all).
# Excluding a chain is for the known label-switching mode-trap, NOT for
# hiding a sampler discrepancy -- record any exclusion in the PR.
NP_CHAINS: list[int] | None = None
STAN_CHAINS: list[int] | None = None


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def read_chains(path: Path, cols: list[str]) -> np.ndarray:
    """Read the given columns and reshape to (n_kept_chains, NUM_DRAWS, n_cols),
    relying on the RUNBOOK layout: chains concatenated in row order, and
    applying the per-side chain subset."""
    t = pq.read_table(path, columns=cols)
    assert t.num_rows == NUM_CHAINS * NUM_DRAWS, (
        f"{path}: expected {NUM_CHAINS * NUM_DRAWS} rows (chains concatenated in "
        f"order per RUNBOOK), got {t.num_rows}"
    )
    x = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols], axis=1)
    x = x.reshape(NUM_CHAINS, NUM_DRAWS, len(cols))
    keep = NP_CHAINS if path == NUMPYRO_OUT else STAN_CHAINS
    if keep is not None:
        x = x[keep]
    return np.ascontiguousarray(x)


def ess_geyer_batch(x: np.ndarray) -> np.ndarray:
    """Vectorized run_comparison.ess_geyer over x of shape (n_chains, n, k):
    same biased-denominator ACF, same initial-positive-pair-sum truncation.
    Returns per-chain ESS of shape (n_chains, k)."""
    nc, n, k = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    var = xc.var(axis=1)                                   # (nc, k)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, nfft, axis=1)
    num = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n, :]   # lagged sums
    acf = num / (np.arange(n, 0, -1)[None, :, None] * var[:, None, :] + 1e-300)
    ts = np.arange(1, n // 2, 2)
    pairs = acf[:, ts, :] + acf[:, ts + 1, :]              # (nc, P, k)
    neg = pairs < 0
    first_neg = np.where(neg.any(axis=1), neg.argmax(axis=1), pairs.shape[1])
    csum = np.cumsum(pairs, axis=1)
    before = np.take_along_axis(csum, np.maximum(first_neg - 1, 0)[:, None, :], axis=1)[:, 0, :]
    tau = 1.0 + 2.0 * np.where(first_neg > 0, before, 0.0)
    ess = np.maximum(n / tau, 1.0)
    return np.where(var <= 0, float(n), ess)


def split_rhat_batch(x: np.ndarray) -> np.ndarray:
    """Vectorized run_comparison.split_rhat over x of shape (n_chains, n, k)."""
    n = x.shape[1]
    m = n // 2
    halves = np.concatenate([x[:, :m, :], x[:, m:2 * m, :]], axis=0)
    W = halves.var(axis=1, ddof=1).mean(axis=0)
    B = m * halves.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((m - 1) / m * W + B / m) / (W + 1e-300))


def nonconstant(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Columns with real variation in BOTH runs. Relative tolerance, not
    std > 0: a constant column stored as e.g. 3.14 carries ~1e-14 float-noise
    std, and cross-sampler float noise there would make z pure garbage."""
    def ok(x):
        f = x.reshape(-1, x.shape[2])
        return f.std(axis=0) > 1e-12 * (np.abs(f.mean(axis=0)) + 1.0)
    return ok(a) & ok(b)


def z_batch(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """z-scores for mean differences, xa/xb of shape (n_chains, n, k)."""
    fa, fb = (x.reshape(-1, x.shape[2]) for x in (xa, xb))
    se = np.sqrt(fa.var(axis=0, ddof=1) / ess_geyer_batch(xa).sum(axis=0)
                 + fb.var(axis=0, ddof=1) / ess_geyer_batch(xb).sum(axis=0))
    return (fa.mean(axis=0) - fb.mean(axis=0)) / (se + 1e-300)


def verdict(abs_z: np.ndarray, label: str) -> bool:
    """Bonferroni-corrected max-|z| test, as in run_comparison.py: a hard
    |z|>3 cutoff is miscalibrated for the max of N tests (expected max grows
    with N under the null)."""
    n = len(abs_z)
    zmax = float(abs_z.max())
    p_max = 1 - (2 * stats.norm.cdf(zmax) - 1) ** n
    thresh = stats.norm.ppf(1 - 0.025 / n)
    print(f"[{label}] WORST |z|: {zmax:.2f}; P(max >= this | null, {n} tests) = {p_max:.2f}; "
          f"Bonferroni threshold {thresh:.2f}")
    print(f"[{label}] |z|>2: {(abs_z > 2).sum()} (expect ~{0.0455 * n:.1f}); "
          f"|z|>3: {(abs_z > 3).sum()} (expect ~{0.0027 * n:.1f})")
    ok = zmax < thresh
    print(f"[{label}] VERDICT:", "PASS — posteriors statistically indistinguishable" if ok
          else "ATTENTION — check flagged parameters / chain health; "
               "rerun with a fresh seed to test stability before concluding")
    return ok


def chain_health():
    for name, path, keep in (("numpyro", NUMPYRO_OUT, NP_CHAINS), ("stan", STAN_OUT, STAN_CHAINS)):
        x = read_chains(path, ["lp__", "divergent__"])
        labels = keep if keep is not None else range(NUM_CHAINS)
        for i, c in enumerate(labels):
            lp, dv = x[i, :, 0], x[i, :, 1]
            print(f"[{name}] chain {c}: divergent={int(dv.sum())}/{NUM_DRAWS} "
                  f"lp__ std={lp.std():.1f} stuck={lp.std() < 1e-6}")


def compare_scalars() -> np.ndarray:
    common = set(parquet_columns(NUMPYRO_OUT)) & set(parquet_columns(STAN_OUT))
    params = sorted(c for c in common if "[" not in c and c not in SKIP)
    a = read_chains(NUMPYRO_OUT, params)
    b = read_chains(STAN_OUT, params)
    nz = nonconstant(a, b)
    a, b, params = a[:, :, nz], b[:, :, nz], [p for p, keep in zip(params, nz) if keep]

    z = z_batch(a, b)
    ra, rb = split_rhat_batch(a), split_rhat_batch(b)
    ma = a.reshape(-1, len(params)).mean(axis=0)
    mb = b.reshape(-1, len(params)).mean(axis=0)
    print(f"\n{'param':38s} {'mean_np':>11s} {'mean_stan':>11s} {'z':>6s} {'rhat_np':>8s} {'rhat_st':>8s}")
    for i, p in enumerate(params):
        print(f"{p:38s} {ma[i]:11.4f} {mb[i]:11.4f} {z[i]:6.2f} {ra[i]:8.3f} {rb[i]:8.3f}")
    print()
    verdict(np.abs(z), "scalars")
    return z


def compare_latents() -> None:
    cols_a, cols_b = parquet_columns(NUMPYRO_OUT), parquet_columns(STAN_OUT)
    common = sorted(set(cols_a) & set(cols_b))
    latents = [c for c in common if "[" in c]
    only_a = sum("[" in c for c in set(cols_a) - set(cols_b))
    only_b = sum("[" in c for c in set(cols_b) - set(cols_a))
    print(f"\nlatent columns: {len(latents)} shared; {only_a} numpyro-only, "
          f"{only_b} stan-only (Stan's unported debugging leftovers / any "
          f"un-ported transformed params -- see NumpyroModel docstring)")

    names, zs, rhats_a, rhats_b, means_a, means_b, const = [], [], [], [], [], [], []
    for i0 in range(0, len(latents), CHUNK):
        cols = latents[i0:i0 + CHUNK]
        a = read_chains(NUMPYRO_OUT, cols)
        b = read_chains(STAN_OUT, cols)
        nz = nonconstant(a, b)
        const += [c for c, keep in zip(cols, nz) if not keep]
        if nz.any():
            a, b = a[:, :, nz], b[:, :, nz]
            names += [c for c, keep in zip(cols, nz) if keep]
            zs.append(z_batch(a, b))
            rhats_a.append(split_rhat_batch(a))
            rhats_b.append(split_rhat_batch(b))
            means_a.append(a.reshape(-1, a.shape[2]).mean(axis=0))
            means_b.append(b.reshape(-1, b.shape[2]).mean(axis=0))
        done = min(i0 + CHUNK, len(latents))
        print(f"  ... {done}/{len(latents)} latent columns", end="\r", flush=True)
    print()

    z = np.concatenate(zs)
    out = pl.DataFrame({
        "param": names, "z": z,
        "mean_numpyro": np.concatenate(means_a), "mean_stan": np.concatenate(means_b),
        "rhat_numpyro": np.concatenate(rhats_a), "rhat_stan": np.concatenate(rhats_b),
    })
    out.write_parquet(LATENTS_RESULT)
    print(f"per-latent results -> {LATENTS_RESULT}")
    if const:
        print(f"skipped {len(const)} constant column(s), e.g. {const[:5]}")

    order = np.argsort(-np.abs(z))[:20]
    print(f"\nworst 20 of {len(names)} latents:")
    print(f"{'param':38s} {'mean_np':>11s} {'mean_stan':>11s} {'z':>6s} {'rhat_np':>8s} {'rhat_st':>8s}")
    for i in order:
        print(f"{names[i]:38s} {out['mean_numpyro'][int(i)]:11.4f} {out['mean_stan'][int(i)]:11.4f} "
              f"{z[i]:6.2f} {out['rhat_numpyro'][int(i)]:8.3f} {out['rhat_stan'][int(i)]:8.3f}")
    print()
    verdict(np.abs(z), "latents")


def compare():
    chain_health()
    compare_scalars()
    compare_latents()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--np-chains", type=str, default=None,
                    help="comma-separated NumPyro chain indices to keep (default: all)")
    ap.add_argument("--stan-chains", type=str, default=None,
                    help="comma-separated Stan chain indices to keep (default: all)")
    args = ap.parse_args()
    if args.np_chains:
        NP_CHAINS = [int(c) for c in args.np_chains.split(",")]
        print(f"NumPyro chains kept: {NP_CHAINS}")
    if args.stan_chains:
        STAN_CHAINS = [int(c) for c in args.stan_chains.split(",")]
        print(f"Stan chains kept: {STAN_CHAINS}")
    compare()
