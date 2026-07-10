# JAX / NumPyro port of `unity_1.8.stan`

A drop-in NumPyro NUTS sampler for the UNITY 1.8 model, producing the same
posterior as CmdStan with ~8× the sampling throughput on CPU and a clear path
to GPUs. Validated end-to-end against Stan at every level (log density,
gradients, full-length sampling) on both a fast test configuration and the
full published Union3.1+UNITY1.8 selection (2085 SNe).

## Using it

Set the model file in any config:

```yaml
fit_model: "unity_1_8_numpyro.py"
```

`Model.from_config` dispatches `.py` model files to `NumpyroModel`
(`src/unity/models/models.py`), which reuses `StanModel`'s data preparation
and blinding wholesale and writes the same `mcmc_samples.parquet` layout.
Sampler-specific config fields:

| field | default | meaning |
|---|---|---|
| `jax_device` | `null` | JAX platform (`"cpu"`, `"cuda"`); applied via `JAX_PLATFORMS` |
| `chain_method` | `"parallel"` | `parallel` = one XLA device per chain (CPU host is split); `vectorized` = batch chains on one device (GPU); `sequential` |
| `sampling_seed` | `null` | PRNG seed; `null` draws a fresh one |

Chains start from `get_initial_position()` (same inits as Stan) via
`init_to_value` — this matters for `om_w0_wa`, whose tight BAO+CMB prior sits
far from a random initialization.

## Design

- `src/unity/models/jax_unity.py` — the entire Stan target (priors included,
  fully normalized) as one jitted JAX function of the constrained parameters.
  Batched over SNe; the 3×3 MVN in closed form; the comoving-distance Simpson
  integration as a cumulative sum.
- `src/unity/models/unity_1_8_numpyro.py` — every Stan parameter as a
  `numpyro.sample` site with an `ImproperUniform` distribution that defines
  only the support (log_prob 0), plus the whole target as a single
  `numpyro.factor`. The constrained-space posterior is therefore exactly the
  Stan model's **by construction** — bounds-as-priors and truncation
  normalizations are never re-derived. NumPyro's unconstraining transforms
  differ from Stan's (notably the simplex), which changes sampler geometry and
  the meaning of `lp__`, but not the posterior.

## Validation

1. **Pointwise parity vs BridgeStan** (`log_density(propto=False, jacobian=False)`):
   - fast config (829 SNe, 3017 unconstrained params): max rel err **8.7e-15**
     over 100 seeded random points (`gen_parity_reference.py` + `check_parity.py`).
   - full selection (2085 SNe, 6892 params): **9.1e-15** over 20 points.
   - every cosmology model (see below): ≤ **9e-15** over 20 points each
     (`check_cosmo_parity.py all`).
2. **Full-length sampling comparisons** on identical frozen data, Geyer-ESS
   z-tests on every shared scalar posterior mean, Bonferroni-calibrated verdict:
   - fast config, NumPyro vs CmdStan (`run_comparison.py`): PASS
     (76 params; mean z = −0.14 ± 0.12, KS vs N(0,1) p = 0.45).
   - full 2085-SN selection, 3-way — NumPyro vs the pristine-master Stan model
     vs the optimized Stan model (`run_full2085.py`): **all three pairs PASS**
     (worst |z| = 1.80 / 2.11 / 2.02 over 37/37/43 quantities). The full-sample
     comparisons use a no-distance-ladder configuration (H0 structurally
     unconstrained).

`lp__` is deliberately never cross-compared between engines: the two differ by
dropped constants and by the Jacobians of their different simplex transforms.

## Performance (i7-10700K, 8 cores, CPU-only, float64)

All engines adapted to the same median trajectory length (127 leapfrog steps),
so these are apples-to-apples:

| engine | config | wall | median ESS/min |
|---|---|---|---|
| NumPyro, 4 parallel chains | 2500+750 draws/chain | 66.8 min | **16.7** |
| optimized Stan, 2 parallel chains | 1250+750 | 246 min | 2.0 |
| baseline Stan, 2 parallel chains | 1250+750 | 504 min | 1.0 |

Per-gradient on the fast config: 1.48 ms (`value_and_grad`) = 9.3× the
optimized Stan model, 21.5× baseline. On GPU (fp64-capable hardware), single
chains are latency-bound; the win is `chain_method: "vectorized"` with many
batched chains.

## Scope and tuning notes

- **Cosmology models: all supported** — 1 (Om), 2 (binned mu), 3 (Om-w),
  4 (q0-j0, Visser eq. 19), 5 (Om-w0-wa CPL incl. the BAO+CMB `multi_normal`
  prior). Model 6 is deprecated upstream and not ported.
- **Photo-z is the one scope gap**: `n_photoz > 0` raises `NotImplementedError`.
- **Warmup**: NumPyro's transforms give different adaptation geometry than
  Stan's — its warmup requirement is its own number. On the full selection,
  warmup 1250 (sufficient for Stan) left a NumPyro chain trapped well below
  the main mode; **use `warmup_iterations >= 2500`** and check per-chain
  `lp__` agreement (a trapped chain shows a per-chain `lp__` mean offset far
  beyond the within-chain spread; `NumpyroModel.fit` warns automatically).
- **Saved columns**: raw parameters, diagnostics, and the cheap transformed
  scalars are emitted; per-SN transformed parameters (`model_mu`, `true_cR`,
  per-SN log-likelihoods, ...) are not yet computed, so downstream PPD/plot
  scripts still need Stan chains. A JAX "generated quantities" pass is the
  planned fix.

## Harness scripts (this directory)

| script | purpose |
|---|---|
| `gen_parity_reference.py` | freeze data + 100 param points + BridgeStan reference |
| `check_parity.py` | acceptance gate: joint lp matches reference to 1e-8 rel |
| `numpyro_model.py` | `--validate` (site wiring vs reference) / `--smoke` (short NUTS) |
| `check_cosmo_parity.py` | per-cosmology BridgeStan sweep (`all` runs every model) |
| `run_comparison.py` | fast-config NumPyro-vs-CmdStan full-length comparison |
| `run_full2085.py` | full-selection NumPyro run + 3-way comparison vs the Stan runs |

`artifacts/` is untracked and regenerable. BridgeStan's `.so` and JAX/XLA
should not share a process (spurious `RESOURCE_EXHAUSTED` observed on macOS);
the harnesses run the two engines in separate phases/processes. When timing
JAX, `jax.block_until_ready` first — pmap dispatch is asynchronous.
