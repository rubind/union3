from pathlib import Path
import pickle
import numpy as np
import multiprocessing

import stan
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import gzip

import union3.utils.helper_functions as helper_functions
from union3.utils.cosmo_functions import get_mu
from union3.utils.file_read import readcol

multiprocessing.set_start_method("fork")

################################################# Get the SALT data ###################################################


################################################# Redshifts for Integration ###################################################


def get_redshifts(redshifts):
    appended_redshifts = np.arange(0.0, 2.51, 0.1)
    tmp_redshifts = np.concatenate((redshifts, appended_redshifts))

    sort_inds = list(np.argsort(tmp_redshifts))
    unsort_inds = [sort_inds.index(i) for i in range(len(tmp_redshifts))]

    tmp_redshifts = np.sort(tmp_redshifts)
    redshifts_sort_fill = np.sort(np.concatenate((tmp_redshifts, 0.5 * (tmp_redshifts[1:] + tmp_redshifts[:-1]))))

    return redshifts_sort_fill, unsort_inds, len(appended_redshifts)


################################################# Redshift Coefficients for Population ###################################################


def plot_coeffs(z_list, redshift_coeffs):
    for log_scale in range(2):
        plt.figure(2)

        sqrtn = int(np.ceil(np.sqrt(len(redshift_coeffs[0]))))

        plt.figure(figsize=(4 * sqrtn, 3 * sqrtn))

        for j in range(len(redshift_coeffs[0])):
            plt.subplot(sqrtn, sqrtn, j + 1)
            plt.plot(z_list, redshift_coeffs[:, j], ".", alpha=0.1, color="b")
            plt.title("Coeff %i" % j)
            if log_scale:
                plt.xscale("log")
        plt.savefig("redshift_coeffs" + "_log" * log_scale + ".pdf", bbox_inches="tight")
        plt.close()

    # assert np.min(np.max(redshift_coeffs, axis = 1) - np.min(redshift_coeffs, axis = 1)) > 0


def get_redshift_coeffs(z_list, p_high_mass, separate_mass_x1c, redshift_coeff_type):
    """redshift_coeff_type could be ("a", 1) or ("a", 3) for a population that varies with a(t)
    redshift_coeff_type could be ("sample", [0.0, 0.4, 1.0]) for a population that is allowed to be different low-z, mid-z, high-z"""

    z_list = np.array(z_list)
    set_list = np.array(the_data["sample_list"])

    if redshift_coeff_type[1].count("."):
        n_z = len(redshift_coeff_type[1:])
    else:
        n_z = int(redshift_coeff_type[1])

    actual_n_x1c_star = n_z * (1 + separate_mass_x1c)

    redshift_coeffs = np.zeros([len(z_list), actual_n_x1c_star], dtype=np.float64)

    if n_z == 1:
        if separate_mass_x1c:
            redshift_coeffs[:, 0] = p_high_mass
            redshift_coeffs[:, 1] = 1 - p_high_mass
        else:
            redshift_coeffs += 1

        plot_coeffs(z_list, redshift_coeffs)
        return redshift_coeffs

    if redshift_coeff_type[0] == "a":
        a_list = 1.0 / (1.0 + np.array(z_list))
        a_nodes = np.linspace(min(a_list) - 1e-5, max(a_list) + 1e-5, n_z)

        for i in range(len(z_list)):
            for j in range(n_z):
                coeffs = np.zeros(n_z, dtype=np.float64)
                coeffs[j] = 1

                ifn = interp1d(a_nodes, coeffs, kind="linear")

                if separate_mass_x1c:
                    redshift_coeffs[i, j] = ifn(a_list[i]) * p_high_mass[i]
                    redshift_coeffs[i, n_z + j] = ifn(a_list[i]) * (1.0 - p_high_mass[i])
                else:
                    redshift_coeffs[i, j] = ifn(a_list[i])

    elif redshift_coeff_type[0] == "sample":
        zs_to_match = np.array([float(item) for item in redshift_coeff_type[1:]])
        print("zs_to_match", zs_to_match)

        for set_ind in np.unique(set_list):
            mean_z = np.mean(z_list[np.where(set_list == set_ind)])

            j = np.argmin(np.abs(zs_to_match - mean_z))
            print("set_ind", set_ind, "mean_z", mean_z, "j", j)

            if separate_mass_x1c:
                redshift_coeffs[:, j] += (set_list == set_ind) * p_high_mass
                redshift_coeffs[:, n_z + j] += (set_list == set_ind) * (1.0 - p_high_mass)
            else:
                redshift_coeffs[:, j] += (set_list == set_ind) * 1.0
    else:
        assert 0, "Unknown redshift_coeff_type " + str(redshift_coeff_type)

    plot_coeffs(z_list, redshift_coeffs)
    return redshift_coeffs


################################################# Binned mu ###################################################


def zcount(z, zmin, zmax):
    return sum((np.array(z) >= zmin) * (np.array(z) < zmax))


def add_zbins(stan_data, cosmo_model):
    # For binned mu

    stan_data["cosmo_model"] = cosmo_model

    print("min, max", stan_data["redshifts"].min(), stan_data["redshifts"].max())
    if stan_data["redshifts"].min() == stan_data["redshifts"].max() or ([2, 6].count(cosmo_model) == 0):
        stan_data["zbins"] = [stan_data["redshifts"][0]]
        stan_data["n_zbins"] = 1
        stan_data["dmu_dbin"] = np.ones([stan_data["n_sne"], stan_data["n_zbins"]], dtype=np.float64)
        stan_data["dmudz_dbin"] = np.zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=np.float64)
        stan_data["mu_const"] = np.zeros(stan_data["n_sne"], dtype=np.float64)
        stan_data["dmu_const_dz"] = np.zeros(stan_data["n_sne"], dtype=np.float64)

        return stan_data

    # if cosmo_model == 2:
    #    zbins = np.exp(np.linspace(np.log(stan_data["redshifts"].min()*0.999),
    #                               np.log(stan_data["redshifts"].max()*1.001), 30))
    # else:
    assert cosmo_model == 6 or cosmo_model == 2
    zsort = np.sort(stan_data["redshifts"])

    print("zsort", zsort[-10:])

    zbins = [zsort[-1] * 1.001]
    step = 10
    minstepsize = 0.1
    min_sn_bin = 10
    ind = -1 - min_sn_bin
    z_cutoff_for_05 = 0.8

    while step > minstepsize:
        step = zbins[0] - zsort[ind]
        minstepsize = ((zbins[0] + zsort[ind]) * 0.5 > z_cutoff_for_05) * 0.05 + 0.05

        if step > minstepsize:
            zbins = [zsort[ind]] + zbins
            ind -= min_sn_bin

    print("zbins high z", zbins)

    zbins = np.concatenate(
        (
            np.linspace(0.05, z_cutoff_for_05, int(0 + np.around(z_cutoff_for_05 / 0.05))),
            np.linspace(z_cutoff_for_05, zbins[0], int(np.around((zbins[0] - z_cutoff_for_05) / 0.1)) + 1)[1:-1],
            zbins,
        )
    )

    zbins = np.array(zbins)

    print("zbins", zbins, list(zbins))

    stan_data["zbins"] = zbins

    stan_data["n_zbins"] = len(stan_data["zbins"])

    f = open("zbins.txt", "w")
    for zbin in stan_data["zbins"]:
        f.write(str(zbin) + "\n")
    f.close()

    plt.figure()
    plt.hist(stan_data["redshifts"], bins=20)
    plt.plot(stan_data["zbins"], [100] * stan_data["n_zbins"], ".", color="k")
    plt.yscale("log")
    plt.savefig("redshift_binning.pdf")
    plt.close()

    stan_data["dmu_dbin"] = np.zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=np.float64)
    stan_data["dmudz_dbin"] = np.zeros([stan_data["n_sne"], stan_data["n_zbins"]], dtype=np.float64)

    for j in range(stan_data["n_zbins"]):
        nodes = np.zeros(stan_data["n_zbins"], dtype=np.float64)
        nodes[j] = 1.0

        if cosmo_model == 6:
            assert 0, "Deprecated for now!"

            minz = min(stan_data["redshifts"]) * 0.999
            ifn = interp1d(
                np.concatenate(([0, minz], stan_data["zbins"])), np.concatenate(([0, minz], nodes)), kind="cubic"
            )
        else:
            assert cosmo_model == 2
            ifn = interp1d(np.concatenate(([0], stan_data["zbins"])), np.concatenate(([-1], nodes)), kind="quadratic")

        for i in range(stan_data["n_sne"]):
            stan_data["dmu_dbin"][i, j] = ifn(stan_data["redshifts"][i])
            stan_data["dmudz_dbin"][i, j] = (
                ifn(stan_data["redshifts"][i] + 0.001) - ifn(stan_data["redshifts"][i])
            ) / 0.001

    if cosmo_model == 6:
        stan_data["mu_const"] = np.zeros(stan_data["n_sne"], dtype=np.float64)
    else:
        stan_data["mu_const"] = get_mu(
            z_list=stan_data["redshifts"],
            cosmo=dict(model="flatLCDM", O_m=0.3, O_k=0.0, h=0.7),
            z_helio_list=stan_data["zhelio"],
        )
        stan_data["dmu_const_dz"] = 1000 * (
            get_mu(
                z_list=stan_data["redshifts"] + 0.001,
                cosmo=dict(model="flatLCDM", O_m=0.3, O_k=0.0, h=0.7),
                z_helio_list=stan_data["zhelio"],
            )
            - stan_data["mu_const"]
        )

    plt.figure(figsize=(8, 20))
    for i in range(len(stan_data["dmu_dbin"][0])):
        plt.plot(stan_data["redshifts"], stan_data["dmu_dbin"][:, i] + i * 2, ".")
        plt.axhline(i * 2, color="k")
    plt.xlabel("Redshift")
    plt.ylabel("$\mu(z) - $ Fiducial")
    plt.yticks([])

    plt.savefig("dmu_dbin.pdf", bbox_inches="tight")
    plt.close()

    return stan_data


################################################# Init FN ###################################################


def init_fn():
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


################################################# Main Program ###################################################

# inputfl = sys.argv[1]
here = Path(__file__).parent
inputfl = (
    here
    / "data"
    / "inputs_Amanullah10_CNIa02_CSP_CalanTololo_CfA1_CfA2_CfA3_CfA4_DES3_Deep_DES3_Shallow_ESSENCE_Foundation_LOSS_MCT_NB99_Pan-STARRS_Riess07_SDSS_SNLS_SuzukiRubin_Tonry03_LSQ+LCO_LSQ_knop03_Krisciunas.pickle"
)
print(
    "cosmo_model: 1 for Om, 2 for binned mu, 3 for Omega_m-w, 4 for q0-j0, 5 for Omega_m-w0-wa, 6 for binned mu with comoving interpolation"
)
# cosmo_model = int(sys.argv[2])
cosmo_model = 1

(the_data, stan_data, params) = pickle.load(gzip.open(inputfl, "rb"))


stan_data = add_zbins(stan_data, cosmo_model)

print("nzadd ", stan_data["nzadd"])
# print stan_data['n_sne']
# print stan_data['n_samples']
# print stan_data['sample_list'].shape
# print stan_data['redshift'].shape
# print stan_data['obs_mBx1c']
# print stan_data['obs_mBx1c_cov'].shape


if stan_data["do_blind"]:
    print("Blinding!")
    # Blind H0

    [zblind, mublind, NA] = readcol(params["blinding_fl"].replace("$UNITY", os.environ["UNITY"]), "fff")
    mublindfn = interp1d(zblind, mublind, kind="linear")
    # dmublinddOmfn = interp1d(zblind, dmublinddOm, kind = 'linear')

    target_distmod = mublindfn(stan_data["redshifts"])
    inds = np.where(stan_data["distmod"] > 0)
    med_offset = np.median(target_distmod[inds] - stan_data["distmod"][inds])
    stan_data["distmod"] += med_offset

    # There are two phases of Hubble-flow blinding:
    # -Making the best-fit Om = 0.3
    # -Bringing all samples into alignment with -19.1 given Om = 0.3

    for iter_count in range(2):
        muobs = (
            stan_data["obs_mBx1c"][:, 0]
            + 0.14 * stan_data["obs_mBx1c"][:, 1]
            - 3.1 * stan_data["obs_mBx1c"][:, 2]
            - -19.1
        )
        H_resid = muobs - mublindfn(stan_data["redshifts"])
        dmuobs = np.sqrt(
            0.15**2.0
            + stan_data["obs_mBx1c_cov"][:, 0, 0]
            + 0.14**2.0 * stan_data["obs_mBx1c_cov"][:, 1, 1]
            + 3.1**2.0 * stan_data["obs_mBx1c_cov"][:, 2, 2]
        )  # Doesn't have to be exact

        for sample_ind in range(stan_data["n_samples"]):
            inds = np.where((the_data["sample_list"] == sample_ind) * (stan_data["redshifts"] >= 0.01))

            if len(inds[0]) > 0:
                med_HR = np.median(H_resid[inds])

                inds = np.where((the_data["sample_list"] == sample_ind))

                for SN_ind in inds[0]:
                    stan_data["obs_mBx1c"][SN_ind, 0] -= med_HR
                    the_data["mB_list"][SN_ind] -= med_HR

                if iter_count > 0:
                    assert abs(med_HR) < 1e-3

        """
        jmat = zeros([stan_data["n_sne"], 2], dtype=float64)
        jmat[:,0] = 1.
        jmat[:,1] = dmublinddOmfn(stan_data["redshifts"])
        wmat = diag(dmuobs**-2.)

        bestvals = dot(
            linalg.inv(dot(transpose(jmat), dot(wmat, jmat))),
            dot(transpose(jmat), dot(wmat, muobs - mublindfn(stan_data["redshifts"])))
            )
        
        
        stan_data["obs_mBx1c"][:,0] -= bestvals[0] + bestvals[1]*dmublinddOmfn(stan_data["redshifts"])
        the_data["mB_list"] = array(the_data["mB_list"]) - bestvals[0] - bestvals[1]*dmublinddOmfn(stan_data["redshifts"])

        if i > 0:
            print(bestvals)
            assert all(abs(bestvals) < 1e-3)
            print("Blinding passed!")
        """


else:
    print("Not Blinding!")
    # assert os.environ["REALLYUNBLIND"] == "1"


print("Running...")

stan_file = here / "models" / "stan_code_H0_1.8.txt"
stan_model = stan_file.read_text()

posterior = stan.build(stan_model, data=stan_data)
fit = posterior.sample(
    num_samples=10,  # params["iter"],
    num_warmup=10,
    num_chains=params["chains"],
    # n_jobs=params["n_jobs"],
    # refresh=10,
    init=[init_fn() for _ in range(params["chains"])],
    # sample_file=params["sample_file"],
)

samples = fit.to_frame()
samples.to_parquet("samples.parquet")


# try:
#     fit_params = helper_functions.filter_fit_params(
#         fit_params, "MB", params["chains"], params["iter"] / 2
#     )  # burns the first half of the chain, so iter/2
# except:
#     print("Couldn't filter bad chains! One or more chains may be bad!")

# # summarize_parameters(fit_params)

# helper_functions.write_latent_variables(the_data=the_data, stan_data=stan_data, fit_params=fit_params)


# del_keys = []
# for key in fit_params:
#     sh = np.array(fit_params[key].shape)

#     if np.any(sh[1:] > params["max_params_to_save"]):
#         print(key, " is too big to save!", sh)
#         del_keys.append(key)

# print("del_keys", del_keys)
# for key in del_keys:
#     del fit_params[key]


# pickle.dump(fit_params, gzip.open("samples.pickle", "wb"))


# print("I hope you have a log file:")

# try:
#     print(fit.stansummary(digits_summary=5))
# except:
#     print("Couldn't print fit! Something is very wrong!")
