"""NumPyro model for unity_1.8, wrapping the validated JAX log density.

Design: every parameter is a `numpyro.sample` site with an ImproperUniform
distribution that only defines its support (and hence NumPyro's unconstraining
transform) — log_prob contributes 0. The ENTIRE Stan target, priors included,
is then added as one `numpyro.factor` using jax_unity.make_logdensity (validated
to 8.7e-15 against BridgeStan). The constrained-space posterior is therefore
exactly the Stan model's; nothing about bounds-as-priors or truncation
normalization is re-derived. NumPyro's transforms (sigmoid for intervals, exp
for lower bounds, stick-breaking for the simplex) may parameterize the
unconstrained space differently than Stan's — that changes sampler geometry,
not the posterior.

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
import numpyro.distributions as dist
from numpyro.distributions import constraints

sys.path.insert(0, str(Path(__file__).parent))
from jax_unity import make_logdensity


def param_spec(data):
    """(name, support constraint, event shape) for every Stan parameter,
    mirroring the parameters block of unity_1.8.stan."""
    n_samples = int(data["n_samples"])
    n_MB = n_samples if int(data["MB_by_sample"]) else 1
    n_sne = int(data["n_sne"])
    n_calib = int(data["n_calib"])
    n_x1c_star = int(data["n_x1c_star"])
    n_zbins = int(data["n_zbins"])
    n_photoz = int(data["n_photoz"])

    iv = constraints.interval
    spec = [
        ("MB_slow", constraints.real, (n_MB,)),
        ("MB_fast_minus_slow", iv(-1.0, 1.0), ()),
        ("H0", iv(50.0, 100.0), ()),
        ("alpha_angle_fast", iv(0.0, 0.35), ()),
        ("alpha_angle_slow", iv(0.0, 0.35), ()),
        ("beta_angle_blue", iv(-1.4, 1.4), ()),
        ("beta_angle_red_low", iv(0.0, 1.4), ()),
        ("beta_angle_red_high", iv(0.0, 1.4), ()),
        ("step_mass", iv(9.5, 10.5), ()),
        ("delta_0", constraints.real, ()),
        ("delta_h", iv(0.0, 1.0), ()),
        ("delta_h_cluster", iv(0.0, 1.0), ()),
        ("Om", iv(0.0, 1.0), ()),
        ("wDE", iv(-2.0, 0.0), ()),
        ("waDE", iv(-5.0, 5.0), ()),
        ("q0", iv(-2.0, 2.0), ()),
        ("j0", iv(-5.0, 5.0), ()),
        ("mu_zbins", constraints.real, (n_zbins,)),
        ("sigma_int", iv(0.01, 0.3), (n_samples,)),
        ("sigma_int_fast", iv(0.01, 0.1), ()),
        ("sigma_int_calibrator", iv(0.01, 0.3), ()),
        ("mBx1c_int_variance", constraints.simplex, (3,)),
        ("true_x1", constraints.real, (n_sne,)),
        ("true_cB", constraints.real, (n_sne,)),
        ("true_cR_unit", constraints.greater_than(-0.25), (n_sne,)),
        ("x1_star_fast", iv(-5.0, -0.5), ()),
        ("R_x1_fast", iv(0.1, 2.0), ()),
        ("x1_star_slow", iv(-0.5, 5.0), ()),
        ("R_x1_slow", iv(0.1, 2.0), ()),
        ("frac_x1_slow", iv(0.01, 0.99), (n_x1c_star,)),
        ("c_star_slow", iv(-0.5, 0.5), ()),
        ("c_star_fast", iv(-0.5, 0.5), ()),
        ("R_c_slow", iv(0.01, 0.2), ()),
        ("R_c_fast", iv(0.01, 0.2), ()),
        ("tau_c", iv(0.01, 0.2), (n_x1c_star,)),
        ("calibs", constraints.real, (n_calib,)),
        ("outl_frac", iv(0.001, 0.1), ()),
        ("mobs_cuts", iv(14.0, 30.0), (n_samples,)),
        ("mobs_cut_sigmas", iv(0.1, 3.0), (n_samples,)),
        ("outl_mBx1c_uncertainties_mB", iv(0.2, 2.0), ()),
        ("outl_mBx1c_uncertainties_x1", iv(1.0, 10.0), ()),
        ("outl_mBx1c_uncertainties_cB", iv(0.1, 1.0), ()),
        ("outl_mBx1c_uncertainties_cR_unit", iv(1.0, 10.0), ()),
    ]
    if n_zbins == 0:
        spec = [s for s in spec if s[0] != "mu_zbins"]
    if n_photoz > 0:
        spec.append(("dz", iv(-0.6, 0.6), (n_photoz,)))
    return spec


def make_model(data):
    core = make_logdensity(data)
    spec = param_spec(data)

    def model():
        p = {
            name: numpyro.sample(name, dist.ImproperUniform(con, (), shape))
            for name, con, shape in spec
        }
        numpyro.factor("stan_target", core(p))

    return model


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
