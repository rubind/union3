"""Stan vs NumPyro posterior comparison for unity_1.8, driven off the paired
configs src/unity/configs/unity_1_8_comparison_{stan,numpyro}.yml.

Both configs run the full published Union3.1+UNITY1.8 selection (2085 SNe,
no distance ladder -> H0 unconstrained -> safe to run unblinded), 4 parallel
chains x 5000 draws each (same chain count on both so wall-clock is directly
comparable). Unblind flags are runtime-only per policy -- never put them in a
config file -- so run the two samplers first, one at a time, timing each:

    time uv run unity --base unity_1_8_comparison_stan.yml --blinding none --really_unblind true
    time uv run unity --base unity_1_8_comparison_numpyro.yml --blinding none --really_unblind true

then:

    uv run python scripts/compare_stan_numpyro.py

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
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "numpyro_port"))
from run_comparison import ess_geyer, split_rhat  # noqa: E402

ROOT = HERE.parent
STAN_OUT = ROOT / "output" / "unity_1_8_comparison" / "stan" / "mcmc_samples.parquet"
NUMPYRO_OUT = ROOT / "output" / "unity_1_8_comparison" / "numpyro" / "mcmc_samples.parquet"
NUM_CHAINS = 4
NUM_DRAWS = 5000
SKIP = {"lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__",
        "divergent__", "energy__", "chain__", "iter__", "draw__"}


def with_chain_col(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    assert df.height == NUM_CHAINS * NUM_DRAWS, (
        f"{path}: expected {NUM_CHAINS * NUM_DRAWS} rows (chains concatenated in "
        f"order per RUNBOOK), got {df.height}"
    )
    return df.with_columns(pl.Series("chain__", np.repeat(np.arange(NUM_CHAINS), NUM_DRAWS)))


def compare():
    a = with_chain_col(NUMPYRO_OUT)
    b = with_chain_col(STAN_OUT)

    for name, df in (("numpyro", a), ("stan", b)):
        for c in sorted(df["chain__"].unique().to_list()):
            sub = df.filter(pl.col("chain__") == c)
            lp = sub["lp__"].to_numpy()
            print(f"[{name}] chain {int(c)}: divergent={int(sub['divergent__'].sum())}/{sub.height} "
                  f"lp__ std={lp.std():.1f} stuck={lp.std() < 1e-6}")

    params = [c for c in a.columns
              if c in b.columns and c not in SKIP and (a[c].std() or 0) > 0]
    print(f"\n{'param':38s} {'mean_np':>11s} {'mean_stan':>11s} {'z':>6s} {'rhat_np':>8s} {'rhat_st':>8s}")
    worst = []
    for p in sorted(params):
        xa, xb = a[p].to_numpy(), b[p].to_numpy()
        ca = [a.filter(pl.col("chain__") == c)[p].to_numpy() for c in a["chain__"].unique()]
        cb = [b.filter(pl.col("chain__") == c)[p].to_numpy() for c in b["chain__"].unique()]
        se = np.sqrt(xa.var(ddof=1) / sum(ess_geyer(c) for c in ca)
                     + xb.var(ddof=1) / sum(ess_geyer(c) for c in cb))
        z = (xa.mean() - xb.mean()) / (se + 1e-300)
        ra, rb = split_rhat(ca), split_rhat(cb)
        worst.append((abs(z), p))
        print(f"{p:38s} {xa.mean():11.4f} {xb.mean():11.4f} {z:6.2f} {ra:8.3f} {rb:8.3f}")

    worst.sort(reverse=True)
    # A hard |z|>3 cutoff is miscalibrated for the max of ~N tests (expected max
    # grows with N under the null): use the Bonferroni-corrected threshold and
    # report the null probability of the observed max, as in run_comparison.py.
    zmax, pname = worst[0]
    n = len(worst)
    p_max = 1 - (2 * stats.norm.cdf(zmax) - 1) ** n
    thresh = stats.norm.ppf(1 - 0.025 / n)
    zs = np.array([w[0] for w in worst])
    print(f"\nWORST |z|: {zmax:.2f} ({pname}); P(max >= this | null, {n} tests) = {p_max:.2f}; "
          f"Bonferroni threshold {thresh:.2f}")
    print(f"|z|>2: {(zs > 2).sum()} (expect ~{0.0455 * n:.1f})")
    print("VERDICT:", "PASS — posteriors statistically indistinguishable" if zmax < thresh
          else "ATTENTION — check flagged parameters / chain health above; "
               "rerun with a fresh seed to test stability before concluding")


if __name__ == "__main__":
    compare()
