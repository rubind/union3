"""JAX port of the unity_1.8.stan joint log density.

`make_logdensity(data)` returns a jitted function mapping a constrained-parameter
dict (params_k.json layout) to the fully normalized joint log density WITHOUT the
constraining-transform Jacobian — i.e. BridgeStan's log_density(propto=False,
jacobian=False). Bounded parameters with no sampling statement contribute nothing
(Stan's implicit uniform is improper); bounds become real priors only at the
NumPyro-model layer, not here.

Scope (handoff §4 phase 2): cosmo_model 1 (Om) and 2 (binned mu), no photo-z.
float64 is mandatory and enabled on import.

Validate against the frozen BridgeStan reference:
    uv run python scripts/numpyro_port/jax_unity.py
"""

import json
import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp, ndtr

LOG_2PI = float(np.log(2.0 * np.pi))
LN10 = float(np.log(10.0))
MU_OFFSET = 43.1586133146

# transformed data: 4-Gaussian approximation to the unit exponential
EXP_APPROX_NORM = np.array(
    [0.15038540936467037, 0.2993904768085472, 0.364279051173158, 0.18594506265362443]
)
EXP_APPROX_POS = np.array(
    [0.10329973984501734, 0.41080906196995237, 1.083137332416308, 2.427349566890827]
)
EXP_APPROX_WIDTH = np.array(
    [0.06596419371844692, 0.1910889454034621, 0.45516250820784515, 1.0637414822809306]
)


def norm_lpdf(y, mu, sigma):
    z = (y - mu) / sigma
    return -0.5 * z * z - jnp.log(sigma) - 0.5 * LOG_2PI


def norm_cdf(y, mu, sigma):
    return ndtr((y - mu) / sigma)


def norm_lcdf(y, mu, sigma):
    return log_ndtr((y - mu) / sigma)


def lognormal_lpdf(y, mu, sigma):
    return norm_lpdf(jnp.log(y), mu, sigma) - jnp.log(y)


def mvn3_lpdf(y, mu, cov):
    """Fully normalized MVN log density for batched symmetric 3x3 covariances,
    via the closed-form adjugate (no Cholesky)."""
    r = y - mu
    a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 0, 2]
    d, e, f = cov[..., 1, 1], cov[..., 1, 2], cov[..., 2, 2]
    A = d * f - e * e  # adjugate entries: inv = adj / det
    B = c * e - b * f
    C = b * e - c * d
    D = a * f - c * c
    E = b * c - a * e
    F = a * d - b * b
    det = a * A + b * B + c * C
    r0, r1, r2 = r[..., 0], r[..., 1], r[..., 2]
    quad = (
        A * r0 * r0 + D * r1 * r1 + F * r2 * r2
        + 2.0 * (B * r0 * r1 + C * r0 * r2 + E * r1 * r2)
    ) / det
    return -0.5 * (quad + jnp.log(det) + 3.0 * LOG_2PI)


def make_logdensity(data):
    d = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in data.items()}

    n_sne = int(d["n_sne"])
    n_samples = int(d["n_samples"])
    cosmo_model = int(d["cosmo_model"])
    MB_by_sample = int(d["MB_by_sample"])
    threeD_unexplained = int(d["threeD_unexplained"])
    do_twoalphabeta = int(d["do_twoalphabeta"])
    do_host_mass = int(d["do_host_mass"])
    fix_Om = float(d["fix_Om"])
    n_zbins = int(d["n_zbins"])

    if int(d["n_photoz"]) != 0:
        raise NotImplementedError("photo-z terms not ported (n_photoz > 0)")
    if cosmo_model not in (1, 2):
        raise NotImplementedError(f"cosmo_model {cosmo_model} not ported (only 1, 2)")

    # static data, 0-indexed where Stan is 1-indexed
    sample_idx = np.asarray(d["sample_list"], dtype=int) - 1
    # length n_sne+nzadd, but only the first n_sne map SNe into r_com_sort
    # (Stan indexes r_com_sort[unsort_inds[i] + 1], 1-based)
    unsort = np.asarray(d["unsort_inds"], dtype=int)[:n_sne]
    obs = jnp.asarray(np.asarray(d["obs_mBx1c"], dtype=float))  # (n,3)
    obs_cov = jnp.asarray(np.asarray(d["obs_mBx1c_cov"], dtype=float))  # (n,3,3)
    dcalib = jnp.asarray(np.asarray(d["d_mBx1c_d_calib"], dtype=float))  # (n,3,nc)
    z_fill = jnp.asarray(np.asarray(d["redshifts_sort_fill"], dtype=float))
    redshifts = jnp.asarray(np.asarray(d["redshifts"], dtype=float))
    zhelio = jnp.asarray(np.asarray(d["zhelio"], dtype=float))
    has_distmod = jnp.asarray(np.asarray(d["has_distmod"], dtype=float))
    distmod = jnp.asarray(np.asarray(d["distmod"], dtype=float))
    mass = jnp.asarray(np.asarray(d["mass"], dtype=float))
    mass_err = jnp.asarray(np.asarray(d["mass_err"], dtype=float))
    in_cluster = jnp.asarray(np.asarray(d["in_cluster"], dtype=float))
    redshift_coeffs = jnp.asarray(np.asarray(d["redshift_coeffs"], dtype=float))
    mobs_cut0 = jnp.asarray(np.asarray(d["mobs_cut0"], dtype=float))
    mobs_cut1 = jnp.asarray(np.asarray(d["mobs_cut1"], dtype=float))
    est_mobs_cuts = jnp.asarray(np.asarray(d["est_mobs_cuts"], dtype=float))
    est_mobs_sigmas = jnp.asarray(np.asarray(d["est_mobs_sigmas"], dtype=float))
    if cosmo_model == 2:
        dmu_dbin = jnp.asarray(np.asarray(d["dmu_dbin"], dtype=float))
        mu_const = jnp.asarray(np.asarray(d["mu_const"], dtype=float))
    outl_ln_mean = float(d["outl_frac_prior_lnmean"])
    outl_ln_width = float(d["outl_frac_prior_lnwidth"])

    gnorm = jnp.asarray(EXP_APPROX_NORM)
    gpos = jnp.asarray(EXP_APPROX_POS)
    gwidth = jnp.asarray(EXP_APPROX_WIDTH)
    log_gnorm = jnp.log(gnorm)

    def logdensity(p):
        arr = lambda k: jnp.atleast_1d(jnp.asarray(p[k], dtype=jnp.float64))
        sca = lambda k: jnp.asarray(p[k], dtype=jnp.float64)

        MB_slow = arr("MB_slow")
        sigma_int = arr("sigma_int")
        mobs_cuts = arr("mobs_cuts")
        mobs_cut_sigmas = arr("mobs_cut_sigmas")
        frac_x1_slow = arr("frac_x1_slow")
        tau_c = arr("tau_c")
        calibs = arr("calibs")
        true_x1 = arr("true_x1")
        true_cB = arr("true_cB")
        true_cR_unit = arr("true_cR_unit")
        mBx1c_int_variance = arr("mBx1c_int_variance")
        mu_zbins = arr("mu_zbins") if n_zbins > 0 else jnp.zeros(0)
        Om, H0 = sca("Om"), sca("H0")
        step_mass = sca("step_mass")
        delta_0, delta_h, delta_h_cluster = sca("delta_0"), sca("delta_h"), sca("delta_h_cluster")
        sigma_int_fast = sca("sigma_int_fast")
        sigma_int_calibrator = sca("sigma_int_calibrator")
        MB_fast_minus_slow = sca("MB_fast_minus_slow")
        outl_frac = sca("outl_frac")
        outl_mB = sca("outl_mBx1c_uncertainties_mB")
        outl_x1 = sca("outl_mBx1c_uncertainties_x1")
        outl_cB = sca("outl_mBx1c_uncertainties_cB")
        outl_cR = sca("outl_mBx1c_uncertainties_cR_unit")

        # ---- distance modulus ----
        if cosmo_model == 1:
            Hinv = 1.0 / jnp.sqrt(Om * (1.0 + z_fill) ** 3 + (1.0 - Om))
            seg = (
                (Hinv[0:-2:2] + 4.0 * Hinv[1:-1:2] + Hinv[2::2])
                * (z_fill[2::2] - z_fill[0:-2:2]) / 6.0
            )
            r_com_sort = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
            model_mu = 5.0 * jnp.log10((1.0 + zhelio) * r_com_sort[unsort]) + MU_OFFSET
        else:  # cosmo_model == 2
            model_mu = dmu_dbin @ mu_zbins + mu_const
        model_mu = jnp.where(
            has_distmod == 1.0, distmod + 5.0 * jnp.log10(H0 / 70.0), model_mu
        )

        # ---- standardization coefficients ----
        alpha_fast = jnp.tan(sca("alpha_angle_fast"))
        alpha_slow = jnp.tan(sca("alpha_angle_slow"))
        if do_twoalphabeta == 0:
            beta_B = jnp.tan(sca("beta_angle_red_low"))
            beta_R_low = beta_B
            beta_R_high = beta_B
        else:
            beta_B = jnp.tan(sca("beta_angle_blue"))
            beta_R_low = jnp.tan(sca("beta_angle_red_low"))
            beta_R_high = jnp.tan(sca("beta_angle_red_high"))

        # ---- intrinsic-scatter vectors per sample, (n_samples, 3), dispersion ----
        s_tot_fast = jnp.sqrt(sigma_int**2 + sigma_int_fast**2)
        if threeD_unexplained == 1:
            scale = jnp.sqrt(mBx1c_int_variance) / jnp.array([1.0, 0.14, -3.0])
            sig_vec_fast = s_tot_fast[:, None] * scale[None, :]
            sig_vec_slow = sigma_int[:, None] * scale[None, :]
        else:
            zero2 = jnp.zeros((n_samples, 2))
            sig_vec_fast = jnp.concatenate([s_tot_fast[:, None], zero2], axis=1)
            sig_vec_slow = jnp.concatenate([sigma_int[:, None], zero2], axis=1)

        eye3 = jnp.eye(3)
        e00 = jnp.zeros((3, 3)).at[0, 0].set(1.0)
        cov_fast = (
            obs_cov
            + (sig_vec_fast[sample_idx] ** 2)[:, :, None] * eye3
            + (has_distmod * sigma_int_calibrator**2)[:, None, None] * e00
        )
        cov_slow = (
            obs_cov
            + (sig_vec_slow[sample_idx] ** 2)[:, :, None] * eye3
            + (has_distmod * sigma_int_calibrator**2)[:, None, None] * e00
        )
        cov_outl = obs_cov + (outl_mB**2) * e00

        # ---- host-mass step ----
        p_high_mass = ndtr((mass - step_mass) / mass_err)
        if do_host_mass == 1:
            this_delta_h = jnp.where(
                in_cluster == 1.0, delta_h + (1.0 - delta_h) * delta_h_cluster, delta_h
            )
            p_high_mass_eff = (
                1.9 * (1.0 - this_delta_h) / (1.0 + 0.9 * jnp.exp(0.95 * LN10 * redshifts))
                + this_delta_h
            ) * p_high_mass
        else:
            p_high_mass_eff = jnp.zeros(n_sne)
        beta_R_eff = beta_R_low * (1.0 - p_high_mass_eff) + beta_R_high * p_high_mass_eff

        # ---- per-SN population blends ----
        frac_x1_slow_by_SN = redshift_coeffs @ frac_x1_slow
        tau_c_by_SN = redshift_coeffs @ tau_c
        true_cR = true_cR_unit * tau_c_by_SN

        this_MB_slow = MB_slow[sample_idx] if MB_by_sample == 1 else MB_slow[0]

        x1_star_fast, x1_star_slow = sca("x1_star_fast"), sca("x1_star_slow")
        R_x1_fast, R_x1_slow = sca("R_x1_fast"), sca("R_x1_slow")
        c_star_fast, c_star_slow = sca("c_star_fast"), sca("c_star_slow")
        R_c_fast, R_c_slow = sca("R_c_fast"), sca("R_c_slow")

        common = model_mu + beta_B * true_cB + beta_R_eff * true_cR - delta_0 * p_high_mass_eff
        mean_fast = jnp.stack(
            [
                this_MB_slow + MB_fast_minus_slow + common
                - alpha_fast * (true_x1 - x1_star_fast),
                true_x1,
                true_cB + true_cR,
            ],
            axis=1,
        )
        mean_slow = jnp.stack(
            [
                this_MB_slow + common - alpha_slow * (true_x1 - x1_star_slow),
                true_x1,
                true_cB + true_cR,
            ],
            axis=1,
        )

        # ---- selection-effect pieces ----
        cuts_by_SN = mobs_cuts[sample_idx]
        cut_sigmas_by_SN = mobs_cut_sigmas[sample_idx]
        bm1 = beta_B + mobs_cut1
        mobs_except_fast = (
            this_MB_slow + MB_fast_minus_slow + model_mu + mobs_cut0
            + bm1 * c_star_fast - delta_0 * p_high_mass_eff
        )
        mobs_except_slow = (
            this_MB_slow + model_mu + mobs_cut0 + bm1 * c_star_slow
            - delta_0 * p_high_mass_eff
        )
        mobs_var_fast = (
            cut_sigmas_by_SN**2
            + cov_fast[:, 0, 0] + cov_fast[:, 2, 2] * mobs_cut1**2
            + 2.0 * mobs_cut1 * cov_fast[:, 0, 2]
            + (alpha_fast * R_x1_fast) ** 2 + (bm1 * R_c_fast) ** 2
        )
        mobs_var_slow = (
            cut_sigmas_by_SN**2
            + cov_slow[:, 0, 0] + cov_slow[:, 2, 2] * mobs_cut1**2
            + 2.0 * mobs_cut1 * cov_slow[:, 0, 2]
            + (alpha_slow * R_x1_slow) ** 2 + (bm1 * R_c_slow) ** 2
        )

        cut_slope = beta_R_eff + mobs_cut1  # (n,)
        cut_mean_shift = cut_slope[:, None] * gpos[None, :] * tau_c_by_SN[:, None]
        cut_var_add = (cut_slope[:, None] * gwidth[None, :] * tau_c_by_SN[:, None]) ** 2
        norm_LL_fast = 0.0001 + jnp.sum(
            gnorm[None, :]
            * ndtr(
                (cuts_by_SN[:, None] - (mobs_except_fast[:, None] + cut_mean_shift))
                / jnp.sqrt(mobs_var_fast[:, None] + cut_var_add)
            ),
            axis=1,
        )
        norm_LL_slow = 0.0001 + jnp.sum(
            gnorm[None, :]
            * ndtr(
                (cuts_by_SN[:, None] - (mobs_except_slow[:, None] + cut_mean_shift))
                / jnp.sqrt(mobs_var_slow[:, None] + cut_var_add)
            ),
            axis=1,
        )

        # ---- observation shift: one calibration matvec per SN ----
        shifted_obs = obs + jnp.einsum("sij,j->si", dcalib, calibs)  # no photo-z

        mobs_lcdf_term = log_ndtr(
            (
                cuts_by_SN
                - (shifted_obs[:, 0] + mobs_cut0 + mobs_cut1 * shifted_obs[:, 2])
            )
            / cut_sigmas_by_SN
        )

        # ---- red-color 4-Gaussian mixture ----
        lse_tmploglike_c = logsumexp(
            log_gnorm[None, :]
            + norm_lpdf(true_cR_unit[:, None], gpos[None, :], gwidth[None, :]),
            axis=1,
        )

        # ---- per-SN mixture components ----
        outl_ll = (
            jnp.log(outl_frac)
            + mvn3_lpdf(shifted_obs, 0.5 * (mean_fast + mean_slow), cov_outl)
            + norm_lpdf(true_x1, 0.0, outl_x1)
            + norm_lpdf(true_cB, 0.0, outl_cB)
            + norm_lpdf(true_cR_unit, 0.0, outl_cR)
        )
        inl_fast = (
            jnp.log(1.0 - outl_frac)
            + jnp.log(1.0 - frac_x1_slow_by_SN)
            + mvn3_lpdf(shifted_obs, mean_fast, cov_fast)
            + norm_lpdf(true_cB, c_star_fast, R_c_fast)
            + norm_lpdf(true_x1, x1_star_fast, R_x1_fast)
            + lse_tmploglike_c
            + mobs_lcdf_term
            - jnp.log(norm_LL_fast)
        )
        inl_slow = (
            jnp.log(1.0 - outl_frac)
            + jnp.log(frac_x1_slow_by_SN)
            + mvn3_lpdf(shifted_obs, mean_slow, cov_slow)
            + norm_lpdf(true_cB, c_star_slow, R_c_slow)
            + norm_lpdf(true_x1, x1_star_slow, R_x1_slow)
            + lse_tmploglike_c
            + mobs_lcdf_term
            - jnp.log(norm_LL_slow)
        )

        lp = jnp.sum(logsumexp(jnp.stack([outl_ll, inl_fast, inl_slow], axis=1), axis=1))

        # ---- priors (all ~ statements, fully normalized) ----
        lp += jnp.sum(norm_lpdf(calibs, 0.0, 1.0))
        if cosmo_model not in (2, 6) and n_zbins > 0:
            lp += jnp.sum(norm_lpdf(mu_zbins, 0.0, 1.0))
        lp += jnp.sum(norm_lpdf(MB_slow, -19.0, 0.5))
        lp += norm_lpdf(delta_0, 0.0, 0.2)
        lp += jnp.sum(norm_lpdf(mobs_cuts, est_mobs_cuts, 0.5))
        lp += jnp.sum(norm_lpdf(mobs_cut_sigmas, est_mobs_sigmas, 0.25))
        if fix_Om > 0:
            lp += norm_lpdf(Om, fix_Om, 0.001)
        lp += norm_lpdf(x1_star_fast, -1.0, 2.0)
        lp += norm_lpdf(x1_star_slow, 1.0, 2.0)
        lp += norm_lpdf(R_x1_fast, 1.0, 2.0)
        lp += norm_lpdf(R_x1_slow, 1.0, 2.0)
        lp += norm_lpdf(c_star_slow, -0.1, 0.2)
        lp += norm_lpdf(c_star_fast, -0.1, 0.2)
        lp += jnp.sum(norm_lpdf(tau_c, 0.1, 0.2))
        lp += norm_lpdf(R_c_slow, 0.1, 0.2)
        lp += norm_lpdf(R_c_fast, 0.1, 0.2)
        if do_twoalphabeta == 0:
            lp += norm_lpdf(sca("beta_angle_blue"), 0.0, 1.0)
        lp += norm_lpdf(outl_mB, 0.5, 0.5)
        lp += norm_lpdf(outl_x1, 3.0, 3.0)
        lp += norm_lpdf(outl_cB, 0.5, 0.5)
        lp += norm_lpdf(outl_cR, 10.0, 3.0)
        lp += lognormal_lpdf(outl_frac, outl_ln_mean, outl_ln_width)
        return lp

    return jax.jit(logdensity)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from check_parity import ART, check

    data = json.loads((ART / "data.json").read_text())
    fn = make_logdensity(data)
    ok = check(lambda p: float(fn(p)), label="jax port")
    sys.exit(0 if ok else 1)
