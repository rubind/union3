"""Full-sample (N=2085) NumPyro run + 3-way posterior comparison against the
Stan speedup-replication chains (baseline master model and optimized model,
output/stan_speedup_replication/*/mcmc_samples.parquet).

Protocol matches those Stan runs: 2 chains x 1250 warmup + 750 draws on the
IDENTICAL data. artifacts/full2085/data.json is frozen by an ephemeral driver
outside the repo (the unblind keywords never appear in tracked files; the run
is safe to unblind ONLY because the distance ladder is structurally absent —
asserted below via has_distmod == 0, so H0 is unconstrained).

    uv run python scripts/numpyro_port/run_full2085.py numpyro [seed] [num_warmup] [num_chains]
    uv run python scripts/numpyro_port/run_full2085.py compare

Warmup note: 1250 (the Stan protocol value) left one NumPyro chain trapped
~930 units of lp__ below the main mode on this config (seed 20260710; kept in
artifacts/full2085/trapped_chain_* for the record) — the known UNITY warmup
pathology presenting at a longer scale in NumPyro's parameterization. Pass
num_warmup=2500 on this config (validated 2026-07-10: 4/4 chains converged,
3-way comparison PASS).

`compare` pools every artifacts/full2085/numpyro*/samples.parquet (extra
evidence chains from alternate seeds included automatically) and z-tests each
shared scalar posterior mean, MC error from per-chain Geyer ESS, with the
same Bonferroni-calibrated verdict as run_comparison.py. lp__ is only
cross-compared between the two Stan runs (NumPyro's differs by constants and
simplex-Jacobian terms).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_comparison import ess_geyer, flat_columns, split_rhat  # noqa: E402

FULL = HERE / "artifacts" / "full2085"
REPL = HERE.parent.parent / "output" / "stan_speedup_replication"

N_WARMUP = 1250
N_DRAWS = 750
N_CHAINS = 2
DIAG = {"lp__", "accept_stat__", "stepsize__", "treedepth__", "n_leapfrog__",
        "divergent__", "energy__", "chain__", "iter__", "draw__"}


def load_data():
    data = json.loads((FULL / "data.json").read_text())
    assert int(np.sum(data["has_distmod"])) == 0, \
        "distance ladder present — this driver must only see ladder-free data"
    assert int(data["n_photoz"]) == 0, "photo-z is outside the validated port scope"
    assert int(data["n_sne"]) == 2085, "not the full published selection"
    return data


def run_numpyro(seed=20260710, num_warmup=N_WARMUP, num_chains=N_CHAINS):
    import numpyro

    numpyro.set_host_device_count(num_chains)  # before jax initializes its backend
    import jax

    jax.config.update("jax_enable_x64", True)
    from numpyro.infer import MCMC, NUTS

    from numpyro_model import make_model

    data = load_data()
    model = make_model(data)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=N_DRAWS,
        num_chains=num_chains,
        chain_method="parallel",
        progress_bar=False,
    )
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed),
             extra_fields=("potential_energy", "num_steps", "diverging"))
    # pmap dispatch is async — block on the results or wall time is meaningless
    samples = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    wall = time.time() - t0

    cols = flat_columns(samples)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    cols["lp__"] = -np.asarray(extra["potential_energy"]).reshape(-1)
    cols["divergent__"] = np.asarray(extra["diverging"]).reshape(-1).astype(float)
    cols["n_leapfrog__"] = np.asarray(extra["num_steps"]).reshape(-1).astype(float)
    cols["chain__"] = np.repeat(np.arange(num_chains), N_DRAWS).astype(float)

    name = "numpyro" if seed == 20260710 else f"numpyro_seed{seed}"
    if num_warmup != N_WARMUP:
        name += f"_w{num_warmup}"
    out = FULL / name
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cols).write_parquet(out / "samples.parquet")
    print(f"[numpyro full2085] DONE. wall {wall / 60:.1f} min "
          f"({num_chains} parallel chains), "
          f"divergences {int(cols['divergent__'].sum())}, "
          f"median leapfrog {np.median(cols['n_leapfrog__']):.0f}")
    # a chain trapped in a low-density region (the warmup pathology) shows up
    # as a per-chain lp__ offset far beyond lp__'s own spread (the trapped
    # chain in the w=1250 run sat ~930 below, ~6x the within-chain std;
    # healthy chains differ by well under 1 std)
    lp = cols["lp__"].reshape(num_chains, N_DRAWS)
    means = lp.mean(axis=1)
    spread_limit = 2.0 * lp.std(axis=1).mean()
    print("per-chain lp__ means:", " ".join(f"{m:.1f}" for m in means),
          "— WARNING: chains far apart, likely trapped chain"
          if np.ptp(means) > spread_limit else "")


def numpyro_frame(data):
    """Pool all numpyro runs (chain__ re-numbered) and derive the transformed
    scalars the Stan runs saved: this_MB_slow (= MB_slow, since
    MB_by_sample=0) and the alpha/beta tans."""
    frames, chain_off = [], 0.0
    for d in sorted(FULL.glob("numpyro*")):
        if not (d / "samples.parquet").exists():
            continue
        df = pl.read_parquet(d / "samples.parquet")
        frames.append(df.with_columns(pl.col("chain__") + chain_off))
        chain_off += float(df["chain__"].n_unique())
    df = pl.concat(frames)

    exprs = [
        pl.col("alpha_angle_fast").tan().alias("alpha_fast"),
        pl.col("alpha_angle_slow").tan().alias("alpha_slow"),
        pl.col("beta_angle_red_low").tan().alias("beta_R_low"),
    ]
    if int(data["do_twoalphabeta"]):
        exprs += [pl.col("beta_angle_blue").tan().alias("beta_B"),
                  pl.col("beta_angle_red_high").tan().alias("beta_R_high")]
    else:
        exprs += [pl.col("beta_angle_red_low").tan().alias("beta_B"),
                  pl.col("beta_angle_red_low").tan().alias("beta_R_high")]
    if not int(data["MB_by_sample"]):
        exprs.append(pl.col("MB_slow").alias("this_MB_slow"))
    return df.with_columns(exprs)


def stan_frame(which):
    df = pl.read_parquet(REPL / which / "mcmc_samples.parquet")
    # no chain__ column: 2 chains x 750 draws concatenated (boundary at row
    # 750 verified via the stepsize__ change point during the replication)
    assert df.height == N_CHAINS * N_DRAWS
    return df.with_columns(pl.Series("chain__", np.repeat([0.0, 1.0], N_DRAWS)))


def health(name, df):
    for c in sorted(df["chain__"].unique().to_list()):
        sub = df.filter(pl.col("chain__") == c)
        lp = sub["lp__"].to_numpy()
        td = (f"  max_treedepth_hits={int((sub['treedepth__'] >= 10).sum())}"
              if "treedepth__" in df.columns else "")
        print(f"[{name}] chain {int(c)}: divergent={int(sub['divergent__'].sum())}/{sub.height}"
              f"{td}  lp__ std={lp.std():.1f}  stuck={lp.std() < 1e-6}")


def pair_ztest(label, a, b, compare_lp=False):
    params = sorted(c for c in a.columns
                    if c in b.columns and c not in DIAG
                    and a[c].dtype.is_float()
                    and (a[c].std() or 0) > 0 and (b[c].std() or 0) > 0)
    if compare_lp:
        params = ["lp__"] + params
    print(f"\n=== {label}: {len(params)} shared quantities ===")
    print(f"{'param':38s} {'mean_a':>11s} {'mean_b':>11s} {'z':>6s} {'rhat_a':>7s} {'rhat_b':>7s}")
    worst = []
    for p in params:
        xa, xb = a[p].to_numpy(), b[p].to_numpy()
        ca = [a.filter(pl.col("chain__") == c)[p].to_numpy() for c in a["chain__"].unique()]
        cb = [b.filter(pl.col("chain__") == c)[p].to_numpy() for c in b["chain__"].unique()]
        se = np.sqrt(xa.var(ddof=1) / sum(ess_geyer(c) for c in ca)
                     + xb.var(ddof=1) / sum(ess_geyer(c) for c in cb))
        z = (xa.mean() - xb.mean()) / (se + 1e-300)
        worst.append((abs(z), p))
        print(f"{p:38s} {xa.mean():11.4f} {xb.mean():11.4f} {z:6.2f} "
              f"{split_rhat(ca):7.3f} {split_rhat(cb):7.3f}")

    from scipy import stats

    worst.sort(reverse=True)
    zmax, pname = worst[0]
    n = len(worst)
    p_max = 1 - (2 * stats.norm.cdf(zmax) - 1) ** n
    thresh = stats.norm.ppf(1 - 0.025 / n)
    zs = np.array([w[0] for w in worst])
    print(f"WORST |z|: {zmax:.2f} ({pname}); P(max >= this | null, {n} tests) = {p_max:.2f}; "
          f"Bonferroni threshold {thresh:.2f}; |z|>2: {(zs > 2).sum()} (expect ~{0.0455 * n:.1f})")
    verdict = "PASS" if zmax < thresh else "ATTENTION"
    print(f"VERDICT [{label}]: {verdict}")
    return verdict, zmax, pname


def compare():
    data = load_data()
    np_df = numpyro_frame(data)
    base = stan_frame("baseline")
    opt = stan_frame("optimized")
    health("numpyro", np_df)
    health("stan-baseline", base)
    health("stan-optimized", opt)

    results = [
        pair_ztest("numpyro vs stan-baseline", np_df, base),
        pair_ztest("numpyro vs stan-optimized", np_df, opt),
        pair_ztest("stan-baseline vs stan-optimized", base, opt, compare_lp=True),
    ]
    print("\n=== 3-WAY SUMMARY ===")
    for (verdict, zmax, pname), label in zip(results, (
            "numpyro vs stan-baseline", "numpyro vs stan-optimized",
            "stan-baseline vs stan-optimized")):
        print(f"{label:35s} {verdict:9s} worst |z|={zmax:.2f} ({pname})")


if __name__ == "__main__":
    if sys.argv[1] == "numpyro":
        run_numpyro(*(int(a) for a in sys.argv[2:5]))
    else:
        {"compare": compare}[sys.argv[1]]()
