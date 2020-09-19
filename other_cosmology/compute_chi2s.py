from DavidsNM import miniNM_new, minos
import pickle
import numpy as np
import tqdm
from cosmo_functions import get_mu, get_R, Planck18_CMB_chi2, get_sound_horizon, load_BAO, get_BAO_chi2
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True
from astropy.io import ascii

from adaptive_contour import adaptive_contour
import sys
import os




def chi2fn(P, passdata):
    run_settings = passdata[0]

    if P[1] <= 0 or P[1] > 0.1:
        return 1e10
    if P[2] <= 0 or P[2] > 2:
        return 1e10
    if P[3] < 0:
        return 1e10
    
    if run_settings["model"] == "flatwCDM":
        if P[3] < 0 or P[3] > 1:
            return 1e10
        if P[5] < -10 or P[5] > 0.2:
            return 1e10

        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], w = P[5])
    elif run_settings["model"] == "LCDM" or run_settings["model"] == "flatLCDM":
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4])
    elif (run_settings["model"] == "flatw0wa") or (run_settings["model"] == "w0wa"):
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], w_0 = P[5], w_a = P[6])
    elif (run_settings["model"] == "binnedrho"):
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], zbins = run_settings["zbins"], rhobins = P[5:])
    else:
        assert 0

        
    chi2 = 0.
            
    if run_settings["include_SNe"]:
        model_mu = get_mu(run_settings["z_list"], cosmo)
        resid = run_settings["mu_list"] - model_mu - P[0]
        chi2 += np.dot(resid, np.dot(run_settings["mu_invcov"], resid))
    if run_settings["include_CMB"]:
        chi2 += Planck18_CMB_chi2(cosmo, run_settings["merged_mat"])
    if run_settings["include_BAO"]:
        chi2 += get_BAO_chi2(BAO_data, cosmo)
        if run_settings["include_CMB"] or run_settings["include_O_mh2"]:
            pass
        else:
            chi2 += ((cosmo["O_bhh"] - 0.02239245)/0.00014778988)**2.


        
    if run_settings["include_O_mh2"]:
        assert run_settings["include_CMB"] == 0
        resid = np.array([cosmo["O_bhh"] - 0.02239245, cosmo["O_m"]*cosmo["h"]**2 - 0.1429665])
        Wmat = np.array([[60924495.84687329, 3501402.45650853],
                         [3501402.45650853, 809719.19225482]], dtype=np.float64)

        chi2 += np.dot(resid, np.dot(Wmat, resid))
    
    return chi2


def get_miniscale(run_settings, global_fit):
    miniscale = np.array(run_settings["miniscale_all"])

    if not global_fit:
        for ind in run_settings["fit_cosmo_inds"]:
            miniscale[ind] = 0
    if not run_settings["include_SNe"]:
        for ind in run_settings["fit_SN_inds"]:
            miniscale[ind] = 0
    if run_settings["include_CMB"] or run_settings["include_BAO"]:
        pass
    else:
        for ind in run_settings["fit_BAOCMB_inds"]:
            miniscale[ind] = 0
    return miniscale


def binned_constraints(z_list, mu_list, mu_invcov, zbins):
    f = fits.open(os.environ["UNITY"] + "/other_cosmology/merged_vals_new_samps_base_w_plikHM_TTTEEE_lowl_lowE.fits")
    merged_mat = f[0].data
    f.close()
    
    ministart = [0.0, 0.022, 0.7, 0.3, 0.0] + [0.5]*len(zbins)
    miniscale = [0.01, 0.001, 0.01, 0.01, 0.0] + [1.]*len(zbins)


    run_settings = dict(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "binnedrho", merged_mat = merged_mat, zbins = zbins, include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0)
    
    P, F, Cmat = miniNM_new(ministart = ministart, miniscale = miniscale, chi2fn = chi2fn, passdata = run_settings)

    print(P)
    print(np.sqrt(np.diag(Cmat)))



def get_minos(bestP, bestF, run_settings):
    this_minos = {}
    
    for i in range(1, len(bestP)):
        this_miniscale = get_miniscale(run_settings, global_fit = 1)
        
        if this_miniscale[i] != 0:
            this_miniscale[i] = 0.
            dx = bestP*0.
            dx[i] = 1
            
            minos_result_plus = minos([bestP.tolist()], [this_miniscale.tolist()], chi2fn = chi2fn, dx = 0.01, targetchi2 = bestF + 1., pos = i, minichi2 = bestF, passdata = run_settings)
            print("minos_result_plus", minos_result_plus)
            
            minos_result_minus = minos([bestP.tolist()], [this_miniscale.tolist()], chi2fn = chi2fn, dx = -0.01, targetchi2 = bestF + 1., pos = i, minichi2 = bestF, passdata = run_settings)
            print("minos_result_minus", minos_result_minus)

            this_minos[run_settings["param_names"][i]] = [bestP[i], minos_result_plus[0], minos_result_minus[0]]
    return this_minos



def make_contours(z_list, mu_list, mu_invcov, model):
    f = fits.open(os.environ["UNITY"] + "/other_cosmology/merged_vals_new_samps_base_w_plikHM_TTTEEE_lowl_lowE.fits")
    merged_mat = f[0].data
    f.close()

    if model == "flatwCDM":
        run_settings = dict(contour_xs = np.linspace(0.0001, 0.5, 30),
                            contour_ys = np.linspace(-2., 0., 31),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, x, 0.0, y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0., 0.1]),
                            fit_cosmo_inds = [3, 5],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok", "w"])
        run_separate_contours = 1
    elif model == "LCDM":
        run_settings = dict(contour_xs = np.linspace(0., 1, 45)**1.5,
                            contour_ys = np.linspace(0., 1.5, 45),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, x, 1 - x - y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0.03]),
                            fit_cosmo_inds = [3, 4],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok"])
        run_separate_contours = 1
    elif model == "flatLCDM":
        run_settings = dict(contour_xs = np.linspace(0., 0.5, 30),
                            contour_ys = np.linspace(0.5, 1.0, 31),
                            ministart_fn = lambda x, y : [0, 0.022, y, x, 0.0],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0.0]),
                            fit_cosmo_inds = [2, 3],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok"])
        run_separate_contours = 1
    elif model == "flatw0wa":
        run_settings = dict(contour_xs = np.linspace(-2., 0., 21),
                            contour_ys = np.linspace(-3., 2., 25),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, 0.3, 0.0, x, y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0., 0.1, 0.1]),
                            fit_cosmo_inds = [5, 6],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok", "w0", "wa"])
        run_separate_contours = 0
    elif model == "w0wa":
        run_settings = dict(contour_xs = np.linspace(-2., 0., 21),
                            contour_ys = np.linspace(-3., 2., 25),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, 0.3, 0.0, x, y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0.02, 0.1, 0.1]),
                            fit_cosmo_inds = [5, 6],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok", "w0", "wa"])
        run_separate_contours = 0
    else:
        assert 0


    run_settings.update(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = model, merged_mat = merged_mat)

    all_grids = {"model": model}


    if run_separate_contours:
        run_settings.update(include_SNe = 1, include_CMB = 0, include_BAO = 0, include_O_mh2 = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                         miniscale = get_miniscale(run_settings, global_fit = 1),
                                         passdata = run_settings,
                                         chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["SNe_chi2"] = bestF
        all_grids["SNe_fit"] = bestP
        all_grids["SNe_minos"] = get_minos(bestP, bestF, run_settings)

        run_settings.update(include_SNe = 0, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["CMB_chi2"] = bestF
        all_grids["CMB_fit"] = bestP

        
        run_settings.update(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 1)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["BAO_Omh2_chi2"] = bestF
        all_grids["BAO_Omh2_fit"] = bestP


        run_settings.update(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["BAO_chi2"] = bestF
        all_grids["BAO_fit"] = bestP


    
    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0)
    bestP_all, bestF_all, bestC_all = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                 miniscale = get_miniscale(run_settings, global_fit = 1),
                                                 passdata = run_settings,
                                                 chi2fn = chi2fn, verbose = False)
    print("bestP_all", bestP_all)
    all_grids["Combined_chi2"] = bestF_all
    all_grids["Combined_fit"] = bestP_all
    all_grids["Combined_cmat"] = bestC_all
    all_grids["Combined_minos"] = get_minos(bestP_all, bestF_all, run_settings)


    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0)
    bestP_SNCMB, bestF_SNCMB, bestC_SNCMB = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                       miniscale = get_miniscale(run_settings, global_fit = 1),
                                                       passdata = run_settings,
                                                       chi2fn = chi2fn, verbose = False)
    print("bestP_SNCMB", bestP_SNCMB)
    all_grids["SNCMB_chi2"] = bestF_SNCMB
    all_grids["SNCMB_fit"] = bestP_SNCMB
    all_grids["SNCMB_cmat"] = bestC_SNCMB
    all_grids["SNCMB_minos"] = get_minos(bestP_SNCMB, bestF_SNCMB, run_settings)



    run_settings.update(include_SNe = 0, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0)
    bestP_BAOCMB, bestF_BAOCMB, bestC_BAOCMB = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                          miniscale = get_miniscale(run_settings, global_fit = 1),
                                                          passdata = run_settings,
                                                          chi2fn = chi2fn, verbose = False)
    print("bestP_BAOCMB", bestP_BAOCMB)
    all_grids["BAOCMB_chi2"] = bestF_BAOCMB
    all_grids["BAOCMB_fit"] = bestP_BAOCMB
    all_grids["BAOCMB_cmat"] = bestC_BAOCMB
    all_grids["BAOCMB_minos"] = get_minos(bestP_BAOCMB, bestF_BAOCMB, run_settings)



    
    run_settings.update(include_SNe = 1, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0)
    bestP_SNBAO, bestF_SNBAO, bestC_SNBAO = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                          miniscale = get_miniscale(run_settings, global_fit = 1),
                                                          passdata = run_settings,
                                                          chi2fn = chi2fn, verbose = False)
    print("bestP_SNBAO", bestP_SNBAO)
    all_grids["SNBAO_chi2"] = bestF_SNBAO
    all_grids["SNBAO_fit"] = bestP_SNBAO
    all_grids["SNBAO_cmat"] = bestC_SNBAO
    all_grids["SNBAO_minos"] = get_minos(bestP_SNBAO, bestF_SNBAO, run_settings)
    


    for include_dict, the_name in [[dict(include_SNe = 1, include_CMB = 0, include_BAO = 0, include_O_mh2 = 0), "SNe"],
                                   [dict(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 1), "BAO_Omh2"],
                                   [dict(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0), "BAO"],
                                   [dict(include_SNe = 0, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0), "CMB"]]*run_separate_contours + [[dict(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0), "Combined"]]:
        run_settings.update(include_dict)
        tmp_chi2fn = lambda x, y: miniNM_new(ministart = run_settings["ministart_fn"](x, y), miniscale = get_miniscale(run_settings, global_fit = 0),
                                             passdata = run_settings,
                                             chi2fn = chi2fn, compute_Cmat = False)[1] - all_grids[the_name + "_chi2"]

        all_xyz, grid_x, grid_y, grid_z = adaptive_contour(tmp_chi2fn, x_1D = run_settings["contour_xs"],
                                                           y_1D = run_settings["contour_ys"],
                                                           contour_levels = np.sort(np.array([2.29575, 6.18007, 11.8292] + [1.0, 5.99146]*model.count("w0wa"))), max_depth = max_depth) # Nominal 5!

        all_grids[the_name] = (grid_x, grid_y, grid_z)
        

    
        plt.figure(figsize = (48, 48))
        
        plt.scatter(all_xyz[0], all_xyz[1], c = all_xyz[2], cmap = "nipy_spectral", vmin = 0, vmax = 50)
        for i in range(len(all_xyz[0])):
            plt.text(all_xyz[0][i], all_xyz[1][i], "%.2g" % all_xyz[2][i], size = 2, rotation = 20)
        plt.colorbar()
        
        plt.savefig("adapt_" + the_name + "_" + model + ".pdf")
        plt.close()


    pickle.dump(all_grids, open("all_grids_" + model + "_" + SN_matrix.split(".fits")[0] + "_max=" + str(max_depth) + ".pickle", 'wb'))

    
    
    

print("python compute_chi2s.py mu_mat.fits 4")

SN_matrix = sys.argv[1]
max_depth = int(sys.argv[2])


f = fits.open(SN_matrix)
dat = f[0].data
f.close()

mu_list = dat[1:, 0]
z_list = dat[0, 1:]
mu_invcov = dat[1:, 1:]

BAO_data = load_BAO()


binned_constraints(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, zbins = [0.2, 0.5, 2.0])
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "flatLCDM")#"flatwCDM")#"LCDM")
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "flatwCDM")#"flatwCDM")#"LCDM")
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "w0wa")
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "flatw0wa")#"flatwCDM")#"LCDM")
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "LCDM")#"flatwCDM")#"LCDM")


