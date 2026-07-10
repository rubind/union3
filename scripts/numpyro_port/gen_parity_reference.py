"""Freeze the parity reference for the NumPyro port of unity_1.8.stan.

Writes to scripts/numpyro_port/artifacts/ (untracked):
  - data.json                  frozen model data (public fast config, stochastically
                               blinded, so nothing unblindable touches disk)
  - params_000..099.json       random full-parameter sets (every parameter in the
                               parameters block, within bounds), seeded
  - reference.npz              per point: Stan unconstrained coordinates, log density
                               with and without the constraining-transform Jacobian
                               (both fully normalized, propto=False), and the gradient
                               wrt unconstrained coordinates

The JAX port validates against this via check_parity.py: its joint log density
evaluated at params_k.json must match lp_nojac[k] to ~1e-8 relative.

Run once from the repo root:
    uv run python scripts/numpyro_port/gen_parity_reference.py
Requires the numpyro extra: uv sync --extra numpyro
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ART = HERE / "artifacts"
REPO = HERE.parent.parent
STAN_FILE = REPO / "src" / "unity" / "models" / "unity_1.8.stan"

N_POINTS = 100
SEED0 = 20260710


def build_data():
    from cmdstanpy import write_stan_json

    from unity import Config, Data, Model

    c = Config(base="union31_H0_snOnly_fastRuns.yml")
    d = Data.from_config(c)
    m = Model.from_config(c)
    m.initialise(d)
    m.blind(kind="stochastic")

    n_calib_sne = int(np.asarray(m.data["has_distmod"]).sum())
    print(
        f"n_sne={m.data['n_sne']} n_samples={m.data['n_samples']} "
        f"n_calib={m.data['n_calib']} calibrators={n_calib_sne}"
    )
    # expected post-scrub public-config counts; abort on anything else so a
    # misconfigured environment can't silently freeze the wrong dataset
    assert m.data["n_sne"] == 829 and n_calib_sne == 6, "unexpected public-config counts"
    write_stan_json(str(ART / "data.json"), m.data)


def random_params(data, rng):
    """One full constrained-parameter set, every parameter within its bounds."""
    n_sne = data["n_sne"]
    n_samples = data["n_samples"]
    n_MB = n_samples if data["MB_by_sample"] else 1
    obs = np.array(data["obs_mBx1c"])  # (n_sne, 3)
    u = lambda lo, hi, size=None: rng.uniform(lo, hi, size)
    return {
        "MB_slow": u(-19.2, -19.0, n_MB),
        "MB_fast_minus_slow": u(-0.05, 0.05),
        "H0": u(65, 75),
        "alpha_angle_fast": u(0.1, 0.2),
        "alpha_angle_slow": u(0.1, 0.2),
        "beta_angle_blue": u(1.1, 1.3),
        "beta_angle_red_low": u(1.1, 1.3),
        "beta_angle_red_high": u(1.1, 1.3),
        "step_mass": u(9.9, 10.1),
        "delta_0": u(-0.02, 0.02),
        "delta_h": u(0.4, 0.6),
        "delta_h_cluster": u(0.4, 0.6),
        "Om": u(0.28, 0.32),
        "wDE": u(-1.1, -0.9),
        "waDE": u(-0.1, 0.1),
        "q0": u(-0.6, -0.4),
        "j0": u(-0.1, 0.1),
        "mu_zbins": rng.normal(size=data["n_zbins"]) * 0.05,
        "sigma_int": u(0.08, 0.12, n_samples),
        "sigma_int_fast": u(0.02, 0.05),
        "sigma_int_calibrator": u(0.04, 0.08),
        "mBx1c_int_variance": rng.dirichlet([5.0, 1.0, 1.0]),
        "true_x1": obs[:, 1] + rng.normal(size=n_sne) * 0.1,
        "true_cB": np.clip(obs[:, 2], -0.2, 0.3) + rng.normal(size=n_sne) * 0.01,
        "true_cR_unit": u(0.3, 1.0, n_sne),
        "x1_star_fast": u(-1.5, -1.0),
        "R_x1_fast": u(0.5, 0.8),
        "x1_star_slow": u(0.2, 0.6),
        "R_x1_slow": u(0.5, 0.8),
        "frac_x1_slow": u(0.4, 0.6, data["n_x1c_star"]),
        "c_star_slow": u(-0.1, 0.0),
        "c_star_fast": u(-0.1, 0.0),
        "R_c_slow": u(0.04, 0.08),
        "R_c_fast": u(0.04, 0.08),
        "tau_c": u(0.05, 0.1, data["n_x1c_star"]),
        "calibs": rng.normal(size=data["n_calib"]) * 0.1,
        "outl_frac": u(0.01, 0.03),
        "mobs_cuts": np.array(data["est_mobs_cuts"]) + rng.normal(size=n_samples) * 0.1,
        "mobs_cut_sigmas": u(0.4, 0.6, n_samples),
        "dz": rng.normal(size=data["n_photoz"]) * 0.01,
        "outl_mBx1c_uncertainties_mB": u(0.4, 0.6),
        "outl_mBx1c_uncertainties_x1": u(2.5, 3.5),
        "outl_mBx1c_uncertainties_cB": u(0.4, 0.6),
        "outl_mBx1c_uncertainties_cR_unit": u(8, 10),
    }


def main():
    ART.mkdir(exist_ok=True)
    sys.argv = sys.argv[:1]  # Config() argparses sys.argv

    if not (ART / "data.json").exists():
        build_data()
    data = json.loads((ART / "data.json").read_text())

    import bridgestan
    from cmdstanpy import write_stan_json

    print("compiling with BridgeStan (first run downloads/compiles; takes minutes)...")
    model = bridgestan.StanModel.from_stan_file(str(STAN_FILE), str(ART / "data.json"))
    dim = model.param_unc_num()
    print(f"model ready: {dim} unconstrained parameters")

    theta_unc = np.empty((N_POINTS, dim))
    lp_nojac = np.empty(N_POINTS)
    lp_jac = np.empty(N_POINTS)
    grad_unc = np.empty((N_POINTS, dim))

    for k in range(N_POINTS):
        rng = np.random.default_rng(SEED0 + k)
        pfile = ART / f"params_{k:03d}.json"
        write_stan_json(str(pfile), random_params(data, rng))
        theta_unc[k] = model.param_unconstrain_json(pfile.read_text())
        lp_nojac[k] = model.log_density(theta_unc[k], propto=False, jacobian=False)
        lp_jac[k], grad_unc[k] = model.log_density_gradient(
            theta_unc[k], propto=False, jacobian=True
        )
        if k % 20 == 0:
            print(f"  point {k:3d}: lp_nojac={lp_nojac[k]:.6f}")

    assert np.isfinite(lp_nojac).all() and np.isfinite(lp_jac).all()
    assert np.isfinite(grad_unc).all()

    np.savez_compressed(
        ART / "reference.npz",
        theta_unc=theta_unc,
        lp_nojac=lp_nojac,
        lp_jac=lp_jac,
        grad_unc=grad_unc,
        param_names=np.array(model.param_names(), dtype=object),
        param_unc_names=np.array(model.param_unc_names(), dtype=object),
    )
    print(f"wrote {N_POINTS} points to {ART / 'reference.npz'}")
    print(f"lp_nojac range: [{lp_nojac.min():.2f}, {lp_nojac.max():.2f}]")


if __name__ == "__main__":
    main()
