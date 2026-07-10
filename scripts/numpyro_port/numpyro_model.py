"""Validation/smoke harness for the packaged NumPyro model of unity_1.8.

The model itself lives in the unity package (src/unity/models/unity_1_8_numpyro.py,
wrapping the validated jax_unity log density); this script keeps the parity
checks against the frozen BridgeStan reference in artifacts/.

Validate site wiring against the parity reference (fast):
    uv run python scripts/numpyro_port/numpyro_model.py --validate
NUTS smoke test (~minutes, prints per-iteration diagnostics):
    uv run python scripts/numpyro_port/numpyro_model.py --smoke
"""

import json
import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpyro

sys.path.insert(0, str(Path(__file__).parent))
from unity.models.unity_1_8_numpyro import make_model, param_spec  # noqa: E402


def _reference_params(data, k=0):
    from check_parity import ART

    raw = json.loads((ART / f"params_{k:03d}.json").read_text())
    names = {s[0] for s in param_spec(data)}
    return {k_: jnp.asarray(v, dtype=jnp.float64) for k_, v in raw.items() if k_ in names}


def validate(data):
    """numpyro.infer.util.log_density (sites contribute 0 + our factor) must
    reproduce the BridgeStan jacobian=False reference at every parity point."""
    from numpyro.infer.util import log_density

    from check_parity import ART

    model = make_model(data)
    ref = np.load(ART / "reference.npz", allow_pickle=True)
    lp_ref = ref["lp_nojac"]
    errs = []
    for k in range(len(lp_ref)):
        lp, _ = log_density(model, (), {}, _reference_params(data, k))
        errs.append(abs(float(lp) - lp_ref[k]) / abs(lp_ref[k]))
    errs = np.array(errs)
    print(f"numpyro log_density vs BridgeStan: max rel err {errs.max():.3e} over {len(errs)} points")
    ok = errs.max() < 1e-8
    print("PASS" if ok else "FAIL")
    return ok


def smoke(data, num_warmup=150, num_samples=100, seed=0):
    from numpyro.infer import MCMC, NUTS, init_to_value

    model = make_model(data)
    init = _reference_params(data, 0)
    kernel = NUTS(model, init_strategy=init_to_value(values=init))
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=1, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("potential_energy", "num_steps", "diverging"))
    pe = np.asarray(mcmc.get_extra_fields()["potential_energy"])
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
    s = mcmc.get_samples()
    print(f"\nlp__ (=-PE) mean {-pe.mean():.1f} std {pe.std():.1f} | divergences {div} | "
          f"median leapfrog steps {np.median(steps):.0f}")
    for k in ("Om", "sigma_int_fast", "outl_frac", "step_mass"):
        x = np.asarray(s[k])
        print(f"  {k}: mean {x.mean():.4f} std {x.std():.4f} (moving: {x.std() > 0})")
    return mcmc


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    from check_parity import ART

    data = json.loads((ART / "data.json").read_text())
    if "--smoke" in sys.argv:
        smoke(data)
    else:
        sys.exit(0 if validate(data) else 1)
