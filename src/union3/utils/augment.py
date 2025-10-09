import numpy as np

from union3.plotting.plot import plot_coeffs
from union3.utils.cosmo_functions import get_mu
from scipy.interpolate import interp1d


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

    # plt.figure()
    # plt.hist(stan_data["redshifts"], bins=20)
    # plt.plot(stan_data["zbins"], [100] * stan_data["n_zbins"], ".", color="k")
    # plt.yscale("log")
    # plt.savefig("redshift_binning.pdf")
    # plt.close()

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

    # plt.figure(figsize=(8, 20))
    # for i in range(len(stan_data["dmu_dbin"][0])):
    #     plt.plot(stan_data["redshifts"], stan_data["dmu_dbin"][:, i] + i * 2, ".")
    #     plt.axhline(i * 2, color="k")
    # plt.xlabel("Redshift")
    # plt.ylabel("$\mu(z) - $ Fiducial")
    # plt.yticks([])

    # plt.savefig("dmu_dbin.pdf", bbox_inches="tight")
    # plt.close()

    return stan_data


def get_redshift_coeffs(z_list, p_high_mass, separate_mass_x1c, redshift_coeff_type, the_data):
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
