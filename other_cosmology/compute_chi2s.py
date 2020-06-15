from DavidsNM import miniNM_new
import pickle
import numpy as np
import tqdm
from cosmo_functions import get_mu, get_R, Planck18_CMB_chi2, get_sound_horizon, load_BAO, get_BAO_chi2
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True
from astropy.io import ascii

from adaptive_contour import adaptive_contour
import sys


def get_colors(key):
    if key == "blue":
        return ((0., 83/255., 152/255.),
                (30/255., 104/255., 168/255.),
                (92/255., 140/255., 190/255.))

    if key == "orange":
        return ((243/255., 116/255., 17/255.),
                (248/255., 144/255., 62/255.),
                (250/255., 180/255., 110/255.))

    if key == "green":
        return ((0., 160/255., 52/255.),
                (50/255., 176/255., 86/255.),
                (117/255., 198/255., 126/255.))
              
    if key == "gray":
        return ((71/255., 71/255., 71/255.),
                (119/255., 119/255., 119/255.),
                (178/255., 178/255., 178/255.))

    if key == "teal":
        return ((0/255., 168/255., 168/255.),
                (127/255., 204/255., 189/255.),
                (190/255., 226/255., 210/255.))
    assert 0, key


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
    elif run_settings["model"] == "LCDM":
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4])
    elif run_settings["model"] == "flatw0wa":
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], w_0 = P[5], w_a = P[6])
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




def make_contours(z_list, mu_list, mu_invcov, model):
    f = fits.open("merged_vals_new_samps_base_w_plikHM_TTTEEE_lowl_lowE.fits")
    merged_mat = f[0].data
    f.close()

    if model == "flatwCDM":
        run_settings = dict(contour_xs = np.linspace(0., 0.5, 30),
                            contour_ys = np.linspace(-2., 0., 31),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, x, 0.0, y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0., 0.1]),
                            fit_cosmo_inds = [3, 5],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2])
        run_separate_contours = 1
    elif model == "LCDM":
        run_settings = dict(contour_xs = np.linspace(0., 1, 30),
                            contour_ys = np.linspace(0., 1.5, 45),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, x, 1 - x - y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0.03]),
                            fit_cosmo_inds = [3, 4],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2])
        run_separate_contours = 1
    elif model == "flatw0wa":
        run_settings = dict(contour_xs = np.linspace(-2., 0., 21),
                            contour_ys = np.linspace(-2., 2., 21),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, 0.3, 0.0, x, y],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0., 0.1, 0.1]),
                            fit_cosmo_inds = [5, 6],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2])
        run_separate_contours = 0
    else:
        assert 0


    run_settings.update(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = model, merged_mat = merged_mat)

    all_grids = {}


    if run_separate_contours:
        run_settings.update(include_SNe = 1, include_CMB = 0, include_BAO = 0, include_O_mh2 = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["SNe_chi2"] = bestF
        

        run_settings.update(include_SNe = 0, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["CMB_chi2"] = bestF
    
        
        run_settings.update(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 1)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["BAO_chi2"] = bestF


    
    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0)
    bestP_all, bestF_all, bestC_all = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                 miniscale = get_miniscale(run_settings, global_fit = 1),
                                                 passdata = run_settings,
                                                 chi2fn = chi2fn, verbose = False)
    print("bestP_all", bestP_all)
    all_grids["Combined_chi2"] = bestF_all


    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0)
    bestP_SNCMB, bestF_SNCMB, bestC_SNCMB = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                       miniscale = get_miniscale(run_settings, global_fit = 1),
                                                       passdata = run_settings,
                                                       chi2fn = chi2fn, verbose = False)
    print("bestP_SNCMB", bestP_SNCMB)

    


    for include_dict, the_name in [[dict(include_SNe = 1, include_CMB = 0, include_BAO = 0, include_O_mh2 = 0), "SNe"],
                                   [dict(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 1), "BAO"],
                                   [dict(include_SNe = 0, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0), "CMB"]]*run_separate_contours + [[dict(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0), "Combined"]]:
        run_settings.update(include_dict)
        tmp_chi2fn = lambda x, y: miniNM_new(ministart = run_settings["ministart_fn"](x, y), miniscale = get_miniscale(run_settings, global_fit = 0),
                                             passdata = run_settings,
                                             chi2fn = chi2fn, compute_Cmat = False)[1] - all_grids[the_name + "_chi2"]

        all_xyz, grid_x, grid_y, grid_z = adaptive_contour(tmp_chi2fn, x_1D = run_settings["contour_xs"],
                                                           y_1D = run_settings["contour_ys"],
                                                           contour_levels = np.array([2.29575, 6.18007, 11.8292]), max_depth = max_depth) # Nominal 5!

        all_grids[the_name] = (grid_x, grid_y, grid_z)
        

    
        plt.figure(figsize = (48, 48))
        
        plt.scatter(all_xyz[0], all_xyz[1], c = all_xyz[2], cmap = "nipy_spectral", vmin = 0, vmax = 50)
        for i in range(len(all_xyz[0])):
            plt.text(all_xyz[0][i], all_xyz[1][i], "%.2g" % all_xyz[2][i], size = 2, rotation = 20)
        plt.colorbar()
        
        plt.savefig("adapt_" + the_name + "_" + model + ".pdf")
        plt.close()


            
    if model == "flatwCDM":
        plt.figure(figsize = (5,5))
    elif model == "flatw0wa":
        plt.figure(figsize = (5,5))
    elif model == "LCDM":
        plt.figure(figsize = (5,7.5))
    else:
        assert 0

    if run_separate_contours:
        plt.contourf(all_grids["BAO"][0], all_grids["BAO"][1], all_grids["BAO"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("green"))
        plt.contourf(all_grids["CMB"][0], all_grids["CMB"][1], all_grids["CMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("orange"))
        plt.contourf(all_grids["SNe"][0], all_grids["SNe"][1], all_grids["SNe"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("blue"))
        plt.contourf(all_grids["Combined"][0], all_grids["Combined"][1], all_grids["Combined"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("gray"))
        
        
        plt.contour(all_grids["SNe"][0], all_grids["SNe"][1], all_grids["SNe"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25)
        plt.contour(all_grids["CMB"][0], all_grids["CMB"][1], all_grids["CMB"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, linestyles = "dotted")
        plt.contour(all_grids["BAO"][0], all_grids["BAO"][1], all_grids["BAO"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25, linestyles = "dashed")
    else:
        plt.contourf(all_grids["Combined"][0], all_grids["Combined"][1], all_grids["Combined"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = get_colors("teal"))
        plt.contour(all_grids["Combined"][0], all_grids["Combined"][1], all_grids["Combined"][2], levels = [0, 2.29575, 6.18007, 11.8292], colors = 'k', linewidths = 0.25)

        

    if model == "flatwCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$w$")
        plt_name = "Om-w.pdf"

    elif model == "flatw0wa":
        plt.xlabel("$w_0$")
        plt.ylabel("$w_a$")
        plt_name = "w0-wa.pdf"

    elif model == "LCDM":
        plt.xlabel("$\Omega_m$")
        plt.ylabel("$\Omega_{\Lambda}$")
        plt.plot([0, 1], [1, 0], color = 'k', linewidth = 0.75)
        plt.axes().set_aspect(1.)
        plt.xlim(0, 1)
        plt.ylim(0, 1.5)
        plt_name = "Om-OL.pdf"

    else:
        assert 0

    all_txt = "All: " + str(bestP_all) + " " + str(np.sqrt(np.diag(bestC_all))) + '\n'
    all_txt += "SN+CMB: " + str(bestP_SNCMB) + " " + str(np.sqrt(np.diag(bestC_SNCMB)))
    
    plt.savefig(plt_name, bbox_inches = 'tight', metadata=dict(Keywords = all_txt))

    

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

make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "flatw0wa")#"flatwCDM")#"LCDM")
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "flatwCDM")#"flatwCDM")#"LCDM")
make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = "LCDM")#"flatwCDM")#"LCDM")


