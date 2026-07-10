"""Full-length sampling comparison: NumPyro NUTS vs CmdStan NUTS on the
IDENTICAL frozen data.json (blinded public fast config, 829 SNe).

Mirrors the Stan-vs-Stan replication protocol: 2 chains x 1250 warmup +
750 draws each, then a z-test on every shared scalar posterior mean with
MC error from per-chain Geyer ESS. Pass: all |z| <~ 3.

lp__ is NOT cross-compared: NumPyro's potential energy differs from Stan's
lp__ by dropped constants and by the (point-dependent) Jacobian of the two
engines' different simplex transforms. Parameter posteriors are invariant.

    uv run python scripts/numpyro_port/run_comparison.py numpyro   # ~30-60 min
    uv run python scripts/numpyro_port/run_comparison.py stan      # ~2 h
    uv run python scripts/numpyro_port/run_comparison.py compare
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
ART = HERE / "artifacts"
OUT = ART / "comparison"
STAN_FILE = HERE.parent.parent / "src" / "unity" / "models" / "unity_1.8.stan"

N_WARMUP = 1250
N_DRAWS = 750
N_CHAINS = 2
MAX_FLAT = 20  # flatten vector sites up to this size into name[i] columns


def flat_columns(samples):
    """samples: dict name -> (n_chains, n_draws, ...); returns dict of 1-D columns
    per chain-concatenated draw, Stan CSV naming (1-based brackets)."""
    cols = {}
    for name, x in samples.items():
        x = np.asarray(x)
        flat = x.reshape(x.shape[0] * x.shape[1], -1)
        if flat.shape[1] == 1:
            cols[name] = flat[:, 0]
        elif flat.shape[1] <= MAX_FLAT:
            for j in range(flat.shape[1]):
                cols[f"{name}[{j + 1}]"] = flat[:, j]
    return cols


def run_numpyro():
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpyro
    from numpyro.infer import MCMC, NUTS

    sys.path.insert(0, str(HERE))
    from numpyro_model import make_model

    data = json.loads((ART / "data.json").read_text())
    model = make_model(data)
    mcmc = MCMC(
        NUTS(model),
        num_warmup=N_WARMUP,
        num_samples=N_DRAWS,
        num_chains=N_CHAINS,
        chain_method="sequential",
        progress_bar=False,
    )
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(20260710),
             extra_fields=("potential_energy", "num_steps", "diverging"))
    wall = time.time() - t0

    s = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    cols = flat_columns(s)
    cols["lp__"] = -np.asarray(extra["potential_energy"]).reshape(-1)
    cols["divergent__"] = np.asarray(extra["diverging"]).reshape(-1).astype(float)
    cols["n_leapfrog__"] = np.asarray(extra["num_steps"]).reshape(-1).astype(float)
    cols["chain__"] = np.repeat(np.arange(N_CHAINS), N_DRAWS).astype(float)

    out = OUT / "numpyro"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cols).write_parquet(out / "samples.parquet")
    div = int(cols["divergent__"].sum())
    print(f"[numpyro] DONE. wall {wall / 60:.1f} min, divergences {div}, "
          f"median leapfrog {np.median(cols['n_leapfrog__']):.0f}")


def run_stan():
    from cmdstanpy import CmdStanModel

    build = ART / "stan_build"
    build.mkdir(parents=True, exist_ok=True)
    stan_copy = build / STAN_FILE.name
    shutil.copy(STAN_FILE, stan_copy)  # compile outside src/ to keep the repo clean
    model = CmdStanModel(stan_file=str(stan_copy))
    t0 = time.time()
    fit = model.sample(
        data=str(ART / "data.json"),
        chains=N_CHAINS,
        parallel_chains=N_CHAINS,
        iter_warmup=N_WARMUP,
        iter_sampling=N_DRAWS,
        seed=20260710,
        show_progress=False,
    )
    wall = time.time() - t0

    df = fit.draws_pd()
    # scalar params + vectors up to MAX_FLAT components (drops per-SN arrays);
    # Stan CSV naming (name[i], 1-based) already matches flat_columns()
    from collections import Counter

    param_cols = [c for c in df.columns if not c.endswith("__")]
    counts = Counter(c.split("[")[0] for c in param_cols)
    cols = {c: df[c].to_numpy() for c in param_cols if counts[c.split("[")[0]] <= MAX_FLAT}
    for dcol in ("lp__", "divergent__", "treedepth__", "n_leapfrog__"):
        cols[dcol] = df[dcol].to_numpy()
    cols["chain__"] = np.repeat(np.arange(N_CHAINS), N_DRAWS).astype(float)

    out = OUT / "stan"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cols).write_parquet(out / "samples.parquet")
    div = int(cols["divergent__"].sum())
    print(f"[stan] DONE. wall {wall / 60:.1f} min, divergences {div}")


def ess_geyer(x):
    n = len(x)
    x = x - x.mean()
    if np.allclose(x, 0):
        return n
    acf = np.correlate(x, x, mode="full")[n - 1:] / (np.arange(n, 0, -1) * x.var() + 1e-300)
    tau = 1.0
    for t in range(1, n // 2, 2):
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return max(n / tau, 1.0)


def split_rhat(per_chain):
    halves = []
    for c in per_chain:
        m = len(c) // 2
        halves += [c[:m], c[m:2 * m]]
    halves = np.array(halves)
    W = halves.var(axis=1, ddof=1).mean()
    B = halves.shape[1] * halves.mean(axis=1).var(ddof=1)
    return np.sqrt(((halves.shape[1] - 1) / halves.shape[1] * W + B / halves.shape[1]) / (W + 1e-300))


def compare():
    a = pl.read_parquet(OUT / "numpyro" / "samples.parquet")
    b = pl.read_parquet(OUT / "stan" / "samples.parquet")
    for name, df in (("numpyro", a), ("stan", b)):
        for c in sorted(df["chain__"].unique().to_list()):
            sub = df.filter(pl.col("chain__") == c)
            lp = sub["lp__"].to_numpy()
            print(f"[{name}] chain {int(c)}: divergent={int(sub['divergent__'].sum())}/{sub.height} "
                  f"lp__ std={lp.std():.1f} stuck={lp.std() < 1e-6}")

    skip = {"lp__", "divergent__", "treedepth__", "n_leapfrog__", "chain__"}
    params = [c for c in a.columns
              if c in b.columns and c not in skip and (a[c].std() or 0) > 0]
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
    n_bad = sum(1 for w in worst if w[0] > 3)
    print(f"\nWORST |z|: {worst[0][0]:.2f} ({worst[0][1]}); params |z|>3: {n_bad}/{len(worst)}")
    print("VERDICT:", "PASS — posteriors statistically indistinguishable" if n_bad == 0
          else "ATTENTION — check flagged parameters / chain health above")


if __name__ == "__main__":
    {"numpyro": run_numpyro, "stan": run_stan, "compare": compare}[sys.argv[1]]()
