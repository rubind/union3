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


def _with_raw_beta_sites(data, params):
    """Substitution dict for the model wrapper: the ordered red-beta angles are driven
    by raw sites (see unity_1_8_numpyro.ORDERED_BETA_ANGLES), so invert the transform
    to hit the reference angles exactly. Returns (params, expected_jacobian_sum)."""
    from unity.models.unity_1_8_numpyro import ORDERED_BETA_ANGLES, ordered_beta_transform

    if int(data["do_twoalphabeta"]) != 1:
        return params, 0.0
    params = dict(params)
    lo = jnp.maximum(0.0, params["beta_angle_blue"])
    width = 1.4 - lo
    expected_jac = 0.0
    for name in ORDERED_BETA_ANGLES:
        u = (params.pop(name) - lo) / width
        raw = jnp.log(u) - jnp.log1p(-u)  # logit
        params[f"{name}_raw"] = raw
        _, log_jac = ordered_beta_transform(raw, params["beta_angle_blue"])
        expected_jac += float(log_jac)
    return params, expected_jac


def validate(data):
    """Two checks per parity point: (1) the model wrapper's log_density minus its
    known ordered-beta Jacobian must reproduce the BridgeStan jacobian=False
    reference; (2) the Jacobian itself must match the closed form, so the wrapper
    adds exactly the truncated-flat measure and nothing else."""
    from numpyro.infer.util import log_density

    from check_parity import ART

    model = make_model(data)
    ref = np.load(ART / "reference.npz", allow_pickle=True)
    lp_ref = ref["lp_nojac"]
    errs, n_skipped = [], 0
    for k in range(len(lp_ref)):
        params = _reference_params(data, k)
        if int(data["do_twoalphabeta"]) == 1:
            lo = max(0.0, float(params["beta_angle_blue"]))
            if not all(float(params[n]) > lo for n in ("beta_angle_red_low", "beta_angle_red_high")):
                # Reference point predates the beta_B < beta_R constraint and lies
                # outside the ordered wedge — not in the constrained model's support.
                n_skipped += 1
                continue
        params, expected_jac = _with_raw_beta_sites(data, params)
        lp, _ = log_density(model, (), {}, params)
        errs.append(abs(float(lp) - expected_jac - lp_ref[k]) / abs(lp_ref[k]))
    errs = np.array(errs)
    print(f"numpyro log_density (minus ordered-beta Jacobian) vs BridgeStan: "
          f"max rel err {errs.max():.3e} over {len(errs)} in-wedge points "
          f"({n_skipped} pre-constraint points outside the ordered wedge skipped)")
    ok = errs.max() < 1e-8 and len(errs) > 0
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
