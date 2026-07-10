"""Per-cosmology BridgeStan parity sweep for the JAX port (jax_unity).

Validates every supported cosmo_model branch — om (1), binned_mu (2), om_w (3),
q0_j0 (4), om_w0_wa (5, incl. the BAO+CMB multi_normal prior) — at 20 seeded
random parameter points each, on frozen full-selection no-ladder data
(artifacts/cosmo/<name>/data.json for the variants, artifacts/full2085/data.json
for om; frozen by an ephemeral out-of-repo driver, ladder structurally absent).
Acceptance gate: rel err < 1e-8 vs log_density(propto=False, jacobian=False);
expect ~1e-14.

BridgeStan's .so and JAX/XLA cannot share a process on this machine (spurious
RESOURCE_EXHAUSTED), so each cosmology runs in two phases:

    uv run python scripts/numpyro_port/check_cosmo_parity.py bridgestan <name>
    uv run python scripts/numpyro_port/check_cosmo_parity.py jax <name>

or "all" to sweep every cosmology via subprocesses:

    uv run python scripts/numpyro_port/check_cosmo_parity.py all
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ART = HERE / "artifacts"
STAN_FILE = HERE.parent.parent / "src" / "unity" / "models" / "unity_1.8.stan"

sys.path.insert(0, str(HERE))
from gen_parity_reference import random_params  # noqa: E402

N_POINTS = 20
SEED0 = 20260710
COSMOS = ("om", "binned_mu", "om_w", "q0_j0", "om_w0_wa")


def data_dir(name):
    return ART / ("full2085" if name == "om" else f"cosmo/{name}")


def points(data):
    for k in range(N_POINTS):
        yield k, random_params(data, np.random.default_rng(SEED0 + k))


def run_bridgestan(name):
    import bridgestan
    from cmdstanpy import write_stan_json

    ddir = data_dir(name)
    data = json.loads((ddir / "data.json").read_text())
    bs = bridgestan.StanModel.from_stan_file(str(STAN_FILE), str(ddir / "data.json"))
    ref = np.empty(N_POINTS)
    pfile = ddir / "_tmp_params.json"
    for k, params in points(data):
        write_stan_json(str(pfile), params)
        theta = bs.param_unconstrain_json(pfile.read_text())
        ref[k] = bs.log_density(theta, propto=False, jacobian=False)
    pfile.unlink()
    assert np.isfinite(ref).all()
    np.save(ddir / "ref_lp.npy", ref)
    print(f"[{name}] wrote {N_POINTS} BridgeStan reference lps "
          f"(range [{ref.min():.2f}, {ref.max():.2f}])")


def run_jax(name):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from unity.models.jax_unity import make_logdensity

    ddir = data_dir(name)
    data = json.loads((ddir / "data.json").read_text())
    logdens = make_logdensity(data)
    ref = np.load(ddir / "ref_lp.npy")
    errs = np.empty(N_POINTS)
    for k, params in points(data):
        lp = float(logdens({k_: jnp.asarray(v, dtype=jnp.float64) for k_, v in params.items()}))
        errs[k] = abs(lp - ref[k]) / abs(ref[k])
    ok = errs.max() < 1e-8
    print(f"[{name}] cosmo_model={data['cosmo_model']}: max rel err {errs.max():.3e} "
          f"over {N_POINTS} points -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "all":
        ok = True
        for name in COSMOS:
            for phase in ("bridgestan", "jax"):
                r = subprocess.run([sys.executable, __file__, phase, name])
                ok &= r.returncode == 0
        print("SWEEP:", "PASS — all cosmologies match BridgeStan" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    elif cmd == "bridgestan":
        run_bridgestan(sys.argv[2])
    elif cmd == "jax":
        sys.exit(0 if run_jax(sys.argv[2]) else 1)
