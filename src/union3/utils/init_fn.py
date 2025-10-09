################################################# Redshift Coefficients for Population ###################################################


################################################# Binned mu ###################################################


################################################# Init FN ###################################################


import numpy as np


def init_fn(stan_data, the_data) -> dict:
    n_sne = len(the_data["x1_list"])
    n_samples = len(the_data["sample_names"])
    print("n_sne ", n_sne)
    print("n_samples ", n_samples)

    if stan_data["cosmo_model"] == 6:
        zbins_tmp = np.array(stan_data["zbins"])
        mu_init = 43.2 + 5 * np.log10((zbins_tmp - 0.225 * zbins_tmp**2.0) * (1.0 + zbins_tmp))
    # elif stan_data["cosmo_model"] == 6:
    #    zbins_tmp = np.array(stan_data["zbins"])
    #    mu_init = zbins_tmp - 0.225*zbins_tmp**2.
    else:
        mu_init = np.random.normal(size=stan_data["n_zbins"]) * 0.05

    return {
        "MB": np.random.random(size=[(n_samples - 1) * stan_data["MB_by_sample"] + 1]) * 0.2 - 19.1,
        "MB_slow": np.random.random(size=[(n_samples - 1) * stan_data["MB_by_sample"] + 1]) * 0.2 - 19.1,
        "MB_fast_minus_slow": np.random.random() * 0.1,
        "Om": 0.3,
        "H0": np.random.random() * 5 + 70.0,
        "wDE": -1.01,
        "mu_zbins": mu_init,
        "alpha_angle": np.arctan(np.random.random() * 0.2),
        "alpha_angle_fast": np.arctan(np.random.random() * 0.2),
        "alpha_angle_slow": np.arctan(np.random.random() * 0.2),
        "beta_angle_blue": np.arctan(np.random.random() * 0.5 + 2.5),
        "beta_angle_blue_fast": np.arctan(np.random.random() * 0.5 + 2.5),
        "beta_angle_blue_slow": np.arctan(np.random.random() * 0.5 + 2.5),
        "beta_angle_red_low": np.arctan(np.random.random() * 0.5 + 2.5),
        "beta_angle_red_high": np.arctan(np.random.random() * 0.5 + 2.5),
        "beta_angle_red_fast": np.arctan(np.random.random() * 0.5 + 2.5),
        "beta_angle_red_slow": np.arctan(np.random.random() * 0.5 + 2.5),
        # "log10_sigma_int": log10(np.random.random(size = n_samples)*0.1 + 0.1),
        "mBx1c_int_variance": [0.9, 0.05, 0.05],
        # "mass_0": 10,
        "delta_0": np.random.random() * 0.05,
        "delta_h": 0.5,
        "step_mass": 10.0,
        "step_width": 0.1,
        "calibs": np.random.normal(size=len(the_data["calib_names"])) * 0.01,
        # "blind_values": [0.]*n_samples,
        "true_cB": np.random.random(size=n_sne) * 0.02 - 0.01 + np.clip(the_data["c_list"] / 2.0, -0.2, 1.0),
        "true_cR_unit": np.random.random(size=n_sne) * 0.5
        + 0.5,  # np.random.random(size = n_sne)*0.01 + clip(the_data["c_list"]/2., 0, 1.0),
        "true_x1": np.random.random(size=n_sne) * 0.2 - 0.1 + the_data["x1_list"],
        "x1_star": np.random.random(size=stan_data["n_x1c_star"]) * 0.5,
        "tau_x1": -np.random.random(size=stan_data["n_x1c_star"]),
        "R_x1": np.random.random(size=stan_data["n_x1c_star"]) * 0.5 + 0.25,
        "x1_star_fast": np.random.random() * 0.5 - 1.25,
        "x1_star_slow": np.random.random() * 0.5,
        "R_x1_fast": np.random.random() * 0.25 + 0.4,
        "R_x1_slow": np.random.random() * 0.25 + 0.4,
        "c_star": -np.random.random(size=stan_data["n_x1c_star"]) * 0.05,
        "c_star_fast": -np.random.random() * 0.05,
        "c_star_slow": -np.random.random() * 0.05,
        "tau_c": np.random.random(size=stan_data["n_x1c_star"]) * 0.05 + 0.02,
        "R_c": np.random.random(size=stan_data["n_x1c_star"]) * 0.05 + 0.02,
        "outl_frac": np.random.random() * 0.02 + 0.01,
        "mobs_cuts": stan_data["est_mobs_cuts"] + np.random.normal(size=n_samples) * 0.1,
        "mobs_cut_sigmas": [0.5] * n_samples,
        "dz": np.random.normal(size=stan_data["n_photoz"]) * 0.01,
    }
