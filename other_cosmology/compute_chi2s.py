from DavidsNM import miniNM_new, minos, save_img
import pickle
import numpy as np
import tqdm
from cosmo_functions import get_mu, get_R, Planck18_CMB_chi2, get_sound_horizon, load_BAO, get_BAO_chi2
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import rcParams
#rcParams['font.family'] = 'serif'
#rcParams['text.usetex'] = True
from astropy.io import ascii

from adaptive_contour import adaptive_contour
import sys
import os




def chi2fn(P, passdata, get_n_data_instead = False):
    run_settings = passdata[0]

    if P[1] <= 0 or P[1] > 0.1:
        return 1e10
    if P[2] <= 0 or P[2] > 2:
        return 1e10
    if P[3] < 0: # Omega_m should never be < 0
        return 1e10

    if run_settings["model"][:4] == "flat":
        if P[3] < 0 or P[3] > 1: # For a flat universe, Omega_m should never be < 0 or > 1
            return 1e10
        
    if run_settings["model"] == "flatwCDM":
        if P[5] < -10 or P[5] > 0.2:
            return 1e10

        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], w = P[5])
        
    elif run_settings["model"] == "LCDM" or run_settings["model"] == "flatLCDM":
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4])

    elif (run_settings["model"] == "flatw0wa") or (run_settings["model"] == "w0wa") or (run_settings["model"] == "flatw0waEDE") or (run_settings["model"] == "w0waEDE") or (run_settings["model"] == "flatw0waEDEfixOm"):
        cosmo = dict(model = run_settings["model"].replace("EDE", ""), O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], w_0 = P[5], w_a = P[6])

        if run_settings["model"].count("EDE") == 0:
            # wa -> -0.302669 - 1.16638 w0 % DE is 1% of matter density at z=1100

            if cosmo["w_a"] > -0.302669 - 1.16638*cosmo["w_0"]:
                return 1e10

    elif (run_settings["model"] == "flatw0waOmh") or (run_settings["model"] == "flatw0waOmhEDE"):
        cosmo = dict(model = "flatw0wa", O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], w_0 = P[5], w_a = P[6])
        if run_settings["model"].count("EDE") == 0:
            # wa -> -0.302669 - 1.16638 w0 % DE is 1% of matter density at z=1100
            if cosmo["w_a"] > -0.302669 - 1.16638*cosmo["w_0"]:
                return 1e10
    elif (run_settings["model"] == "binnedrho"):
        cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], zbins = run_settings["zbins"], rhobins = P[5:])
    else:
        assert 0

        
    chi2 = 0.
    n_data = 0
    
    if run_settings["include_SNe"]:
        model_mu = get_mu(run_settings["z_list"], cosmo)
        resid = run_settings["mu_list"] - model_mu - P[0]
        chi2 += np.dot(resid, np.dot(run_settings["mu_invcov"], resid))
        n_data += len(run_settings["mu_list"])
    if run_settings["include_CMB"]:
        chi2 += Planck18_CMB_chi2(cosmo, run_settings["merged_mat"])
        n_data += 3
    if run_settings["include_H0Ceph"]:
        assert run_settings["include_H0TRGB"] == 0
        #chi2 += ((cosmo["h"]*100. - 72.53)/0.99)**2.
        #chi2 += ((cosmo["h"]*100. - 73.29)/0.90)**2. Leveraging SN Ia spectroscopic similarity to improve the measurement of 
        chi2 += ((cosmo["h"]*100. - 73.17)/0.86)**2. # Small Magellanic Cloud Cepheids Observed with the Hubble Space Telescope Provide a New Anchor for the SH0ES Distance Ladder
        n_data += 1
    if run_settings["include_H0TRGB"]:
        assert run_settings["include_H0Ceph"] == 0
        chi2 += ((cosmo["h"]*100. - 70.39)/1.80)**2. # Status Report on the Chicago-Carnegie Hubble Program (CCHP): Measurement of the Hubble Constant Using the Hubble and James Webb Space Telescopes
        n_data += 1
    if run_settings["include_BAO"]:
        chi2 += get_BAO_chi2(BAO_data, cosmo)
        n_data += len(BAO_data["constraint"])

        if run_settings["include_CMB"] or run_settings["include_O_mh2"]:
            pass
        else:
            chi2 += ((100*cosmo["O_bhh"] - 2.208)/0.052)**2. # Cooke+ 2016, as interpreted through Dark Energy Survey Year 1 Results: A Precise H0 Measurement from DES Y1, BAO, and D/H Data
            #chi2 += ((cosmo["O_bhh"] - 0.02239245)/0.00014778988)**2.
            n_data += 1

        
    if run_settings["include_O_mh2"]:
        assert run_settings["include_CMB"] == 0
        resid = np.array([cosmo["O_bhh"] - 0.02239245, cosmo["O_m"]*cosmo["h"]**2 - 0.1429665])
        Wmat = np.array([[60924495.84687329, 3501402.45650853],
                         [3501402.45650853, 809719.19225482]], dtype=np.float64)

        chi2 += np.dot(resid, np.dot(Wmat, resid))
        n_data += 2
    if run_settings["model"] == "flatw0waEDEfixOm":
        chi2 += ((cosmo["O_m"] - 0.3)/0.001)**2.

    if get_n_data_instead:
        return n_data
    else:
        return chi2


def get_miniscale(run_settings, global_fit, get_n_par_instead = False):
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
    if get_n_par_instead == False:
        return miniscale
    else:
        return np.sum([item != 0 for item in miniscale])

    
def binned_constraints(z_list, mu_list, mu_invcov, zbins, include_BAO):
    f = fits.open(os.environ["UNITY"] + "/other_cosmology/merged_vals_new_samps_base_w_plikHM_TTTEEE_lowl_lowE.fits")
    merged_mat = f[0].data
    f.close()
    
    ministart = [0.0, 0.022, 0.7, 0.3, 0.0] + [0.5]*len(zbins) # The value less than zbins[0] is set by the cosmic sum rule
    miniscale = [0.01, 0.001, 0.01, 0.01, 0.0] + [1.]*len(zbins)

    #         cosmo = dict(model = run_settings["model"], O_bhh = P[1], h = P[2], O_m = P[3], O_k = P[4], zbins = run_settings["zbins"], rhobins = P[5:])

    run_settings = dict(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, miniscale_all = miniscale,
                        model = "binnedrho", merged_mat = merged_mat, zbins = zbins, include_SNe = 1,
                        include_CMB = 1, include_BAO = include_BAO, include_O_mh2 = 0, include_H0Ceph = 0, include_H0TRGB = 0,
                        param_names = ["MB", "O_bhh", "h", "Om", "Ok"] + ["rhobin_%02i" % (i+1) for i in range(len(zbins))])
    
    P, F, Cmat = miniNM_new(ministart = ministart, miniscale = miniscale, chi2fn = chi2fn, passdata = run_settings)

    try:
        Cmat[0,0]
    except:
        return zbins*100

    print("Binned:")
    print(P)
    print("bins", zbins, "include_BAO", include_BAO, "uncertainties", list(np.sqrt(np.diag(Cmat))))
    return np.sqrt(np.diag(Cmat))[-len(zbins):]  #get_minos(P, F, run_settings)
    #print(get_minos(P, F, run_settings))
    #stop_here

def find_best_binned_improvement():
    f = fits.open("mu_mat_Union3_blinded.fits")
    dat = f[0].data
    f.close()
    
    mu_list_U3 = dat[1:, 0]
    
    if np.max(np.abs(mu_list_U3)) < 10:
        print("This looks like LCDM residuals! Adding mu(z)")
        
        mu_list_U3 += get_mu(z_list = dat[0, 1:],
                             cosmo = dict(model = "flatLCDM", O_m = 0.3, O_k = 0.0, h = 0.7))
        
    
    z_list_U3 = dat[0, 1:]
    mu_invcov_U3 = dat[1:, 1:]


    f = fits.open("mu_mat_SeeChangeBlind_mu.fits")
    dat = f[0].data
    f.close()
    
    mu_list_U31 = dat[1:, 0]
    
    if np.max(np.abs(mu_list_U31)) < 10:
        print("This looks like LCDM residuals! Adding mu(z)")
        
        mu_list_U31 += get_mu(z_list = dat[0, 1:],
                              cosmo = dict(model = "flatLCDM", O_m = 0.3, O_k = 0.0, h = 0.7))
        
    
    z_list_U31 = dat[0, 1:]
    mu_invcov_U31 = dat[1:, 1:]

    
    for i in tqdm.trange(100):

        nbins = np.random.randint(3, 9)
        zbins = np.sort(np.random.random(size = nbins)*2.5)

        if min(zbins[1:] - zbins[:-1]) < 0.05:
            pass
        else:
            for include_BAO in [0, 1]:
                union3uncs = binned_constraints(z_list=z_list_U3, mu_list=mu_list_U3, mu_invcov=mu_invcov_U3, zbins=zbins, include_BAO=include_BAO)
                union31uncs = binned_constraints(z_list=z_list_U31, mu_list=mu_list_U31, mu_invcov=mu_invcov_U31, zbins=zbins, include_BAO=include_BAO)
                
                if np.min(union3uncs/union31uncs) > 1.1:
                    print("*"*42)
                print("ZBINS", zbins, "BAO", include_BAO, union3uncs/union31uncs)
                if np.min(union3uncs/union31uncs) > 1.1:
                    print("*"*42)


"""
BAO_data = load_BAO()
find_best_binned_improvement()
stop_here
"""

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
        run_SN_only_contours = 0 # Included in run_separate_contours
        run_inverse_ladder = 0
        run_SNeCMB_and_BAOCMB = 0
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
        run_SN_only_contours = 0 # Included in run_separate_contours
        run_inverse_ladder = 0
        run_SNeCMB_and_BAOCMB = 0
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
        run_SN_only_contours = 0 # Included in run_separate_contours
        run_inverse_ladder = 0
        run_SNeCMB_and_BAOCMB = 0
    elif (model == "flatw0waOmh") or (model == "flatw0waOmhEDE"):
        run_settings = dict(contour_xs = np.linspace(0., 0.5, 30),
                            contour_ys = np.linspace(0.5, 1.0, 31),
                            ministart_fn = lambda x, y : [0, 0.022, y, x, 0.0, -1., 0.],
                            miniscale_all = np.array([0.02, 0.001, 0.01, 0.02, 0.0, 0.2, 0.4]),
                            fit_cosmo_inds = [2, 3],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok", "w0", "wa"])
        run_separate_contours = 0
        run_SN_only_contours = 0
        run_inverse_ladder = 1
        run_SNeCMB_and_BAOCMB = 1
    elif (model == "flatw0wa") or (model == "flatw0waEDE") or (model == "flatw0waEDEfixOm"):
        run_settings = dict(contour_xs = np.linspace(-2., 0.5, 26),
                            contour_ys = np.linspace(-4., 2., 30),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, 0.3, 0.0, x, y],
                            miniscale_all = np.array([0.02, 0.001, 0.05, 0.1, 0., 0.1, 0.2]),
                            fit_cosmo_inds = [5, 6], # inds to set miniscale to zero
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok", "w0", "wa"])
        run_separate_contours = 0
        run_SN_only_contours = 1
        run_inverse_ladder = 1
        run_SNeCMB_and_BAOCMB = 1
        
        if (model == "flatw0waEDEfixOm"):
            run_separate_contours = 0
            run_SN_only_contours = 1
            run_inverse_ladder = 0
            run_SNeCMB_and_BAOCMB = 0
        
    elif (model == "w0wa") or (model == "w0waEDE"):
        run_settings = dict(contour_xs = np.linspace(-2., 0., 21),
                            contour_ys = np.linspace(-3., 2., 25),
                            ministart_fn = lambda x, y : [0, 0.022, 0.7, 0.3, 0.0, x, y],
                            miniscale_all = np.array([0.02, 0.001, 0.05, 0.1, 0.05, 0.1, 0.2]),
                            fit_cosmo_inds = [5, 6],
                            fit_SN_inds = [0],
                            fit_BAOCMB_inds = [1, 2],
                            param_names = ["MB", "O_bhh", "h", "Om", "Ok", "w0", "wa"])
        run_separate_contours = 0
        run_SN_only_contours = 0
        run_inverse_ladder = 1
        run_SNeCMB_and_BAOCMB = 1
    else:
        assert 0


    run_settings.update(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = model, merged_mat = merged_mat)

    all_grids = {"model": model}


    if run_separate_contours or run_SN_only_contours:
        run_settings.update(include_SNe = 1, include_CMB = 0, include_BAO = 0, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                         miniscale = get_miniscale(run_settings, global_fit = 1),
                                         passdata = run_settings,
                                         chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["SNe_chi2"] = bestF
        all_grids["SNe_fit"] = bestP
        all_grids["SNe_minos"] = get_minos(bestP, bestF, run_settings)
        all_grids["SNe_n_data"] = chi2fn(bestP, [run_settings], get_n_data_instead = True)
        all_grids["SNe_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)
    
    if run_separate_contours:    
        run_settings.update(include_SNe = 0, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["CMB_chi2"] = bestF
        all_grids["CMB_fit"] = bestP
        all_grids["CMB_n_data"] = chi2fn(bestP, [run_settings], get_n_data_instead = True)
        all_grids["CMB_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

        
        run_settings.update(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 1, include_H0TRGB = 0, include_H0Ceph = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["BAO_Omh2_chi2"] = bestF
        all_grids["BAO_Omh2_fit"] = bestP
        all_grids["BAO_Omh2_n_data"] = chi2fn(bestP, [run_settings], get_n_data_instead = True)
        all_grids["BAO_Omh2_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)


        run_settings.update(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
        bestP, bestF, NA = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                      miniscale = get_miniscale(run_settings, global_fit = 1),
                                      passdata = run_settings,
                                      chi2fn = chi2fn, compute_Cmat = False, verbose = False)
        print("bestP", bestP)
        all_grids["BAO_chi2"] = bestF
        all_grids["BAO_fit"] = bestP
        all_grids["BAO_n_data"] = chi2fn(bestP, [run_settings], get_n_data_instead = True)
        all_grids["BAO_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

    ##########################################


    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
    bestP_all, bestF_all, bestC_all = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                 miniscale = get_miniscale(run_settings, global_fit = 1),
                                                 passdata = run_settings,
                                                 chi2fn = chi2fn, verbose = False)
    print("bestP_all", bestP_all)
    all_grids["SNeBAOCMB_chi2"] = bestF_all
    all_grids["SNeBAOCMB_fit"] = bestP_all
    all_grids["SNeBAOCMB_cmat"] = bestC_all
    all_grids["SNeBAOCMB_minos"] = get_minos(bestP_all, bestF_all, run_settings)
    all_grids["SNeBAOCMB_n_data"] = chi2fn(bestP_all, [run_settings], get_n_data_instead = True)
    all_grids["SNeBAOCMB_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

        
    ##########################################

    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 1, include_H0Ceph = 0)
    bestP_all, bestF_all, bestC_all = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                 miniscale = get_miniscale(run_settings, global_fit = 1),
                                                 passdata = run_settings,
                                                 chi2fn = chi2fn, verbose = False)
    print("bestP_all", bestP_all)
    all_grids["SNeBAOCMBH0T_chi2"] = bestF_all
    all_grids["SNeBAOCMBH0T_fit"] = bestP_all
    all_grids["SNeBAOCMBH0T_cmat"] = bestC_all
    all_grids["SNeBAOCMBH0T_minos"] = get_minos(bestP_all, bestF_all, run_settings)
    all_grids["SNeBAOCMBH0T_n_data"] = chi2fn(bestP_all, [run_settings], get_n_data_instead = True)
    all_grids["SNeBAOCMBH0T_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

    ##########################################

    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 1)
    bestP_all, bestF_all, bestC_all = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                 miniscale = get_miniscale(run_settings, global_fit = 1),
                                                 passdata = run_settings,
                                                 chi2fn = chi2fn, verbose = False)
    print("bestP_all", bestP_all)
    all_grids["SNeBAOCMBH0C_chi2"] = bestF_all
    all_grids["SNeBAOCMBH0C_fit"] = bestP_all
    all_grids["SNeBAOCMBH0C_cmat"] = bestC_all
    all_grids["SNeBAOCMBH0C_minos"] = get_minos(bestP_all, bestF_all, run_settings)    
    all_grids["SNeBAOCMBH0C_n_data"] = chi2fn(bestP_all, [run_settings], get_n_data_instead = True)
    all_grids["SNeBAOCMBH0C_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

    ##########################################

    run_settings.update(include_SNe = 1, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
    bestP_SNeCMB, bestF_SNeCMB, bestC_SNeCMB = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                          miniscale = get_miniscale(run_settings, global_fit = 1),
                                                           passdata = run_settings,
                                                           chi2fn = chi2fn, verbose = False)
    print("bestP_SNeCMB", bestP_SNeCMB)
    all_grids["SNeCMB_chi2"] = bestF_SNeCMB
    all_grids["SNeCMB_fit"] = bestP_SNeCMB
    all_grids["SNeCMB_cmat"] = bestC_SNeCMB
    all_grids["SNeCMB_minos"] = get_minos(bestP_SNeCMB, bestF_SNeCMB, run_settings)
    all_grids["SNeCMB_n_data"] = chi2fn(bestP_SNeCMB, [run_settings], get_n_data_instead = True)
    all_grids["SNeCMB_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

    ##########################################


    run_settings.update(include_SNe = 0, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
    bestP_BAOCMB, bestF_BAOCMB, bestC_BAOCMB = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                          miniscale = get_miniscale(run_settings, global_fit = 1),
                                                          passdata = run_settings,
                                                          chi2fn = chi2fn, verbose = False)
    print("bestP_BAOCMB", bestP_BAOCMB)
    all_grids["BAOCMB_chi2"] = bestF_BAOCMB
    all_grids["BAOCMB_fit"] = bestP_BAOCMB
    all_grids["BAOCMB_cmat"] = bestC_BAOCMB
    all_grids["BAOCMB_minos"] = get_minos(bestP_BAOCMB, bestF_BAOCMB, run_settings)
    all_grids["BAOCMB_n_data"] = chi2fn(bestP_BAOCMB, [run_settings], get_n_data_instead = True)
    all_grids["BAOCMB_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

    ##########################################

    
    run_settings.update(include_SNe = 1, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0)
    bestP_SNeBAO, bestF_SNeBAO, bestC_SNeBAO = miniNM_new(ministart = run_settings["ministart_fn"](np.mean(run_settings["contour_xs"]), np.mean(run_settings["contour_ys"])),
                                                          miniscale = get_miniscale(run_settings, global_fit = 1),
                                                          passdata = run_settings,
                                                          chi2fn = chi2fn, verbose = False)
    print("bestP_SNeBAO", bestP_SNeBAO)
    all_grids["SNeBAO_chi2"] = bestF_SNeBAO
    all_grids["SNeBAO_fit"] = bestP_SNeBAO
    all_grids["SNeBAO_cmat"] = bestC_SNeBAO
    all_grids["SNeBAO_minos"] = get_minos(bestP_SNeBAO, bestF_SNeBAO, run_settings)
    all_grids["SNeBAO_n_data"] = chi2fn(bestP_SNeBAO, [run_settings], get_n_data_instead = True)
    all_grids["SNeBAO_n_par"] = get_miniscale(run_settings, global_fit = 1, get_n_par_instead = True)

    ##########################################

    
    all_combs_to_run_grid = [[dict(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "SNeBAOCMB"],
                             [dict(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 1), "SNeBAOCMBH0C"],
                             [dict(include_SNe = 1, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 1, include_H0Ceph = 0), "SNeBAOCMBH0T"]]*(model != "flatw0waEDEfixOm")

    if run_separate_contours or run_SN_only_contours:
        all_combs_to_run_grid += [[dict(include_SNe = 1, include_CMB = 0, include_BAO = 0, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "SNe"]]

    if run_separate_contours:
        all_combs_to_run_grid += [[dict(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 1, include_H0TRGB = 0, include_H0Ceph = 0), "BAO_Omh2"],
                                  [dict(include_SNe = 0, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "BAO"],
                                  [dict(include_SNe = 0, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "CMB"]]
    if run_inverse_ladder:
        all_combs_to_run_grid += [[dict(include_SNe = 1, include_CMB = 0, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "SNeBAO"]]

    if run_SNeCMB_and_BAOCMB:
        all_combs_to_run_grid += [[dict(include_SNe = 1, include_CMB = 1, include_BAO = 0, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "SNeCMB"],
                                  [dict(include_SNe = 0, include_CMB = 1, include_BAO = 1, include_O_mh2 = 0, include_H0TRGB = 0, include_H0Ceph = 0), "BAOCMB"]]


        
    for include_dict, the_name in all_combs_to_run_grid:
        
        run_settings.update(include_dict)
        tmp_chi2fn = lambda x, y: miniNM_new(ministart = run_settings["ministart_fn"](x, y), miniscale = get_miniscale(run_settings, global_fit = 0),
                                             passdata = run_settings,
                                             chi2fn = chi2fn, compute_Cmat = False)[1] - all_grids[the_name + "_chi2"]

        all_xyz, grid_x, grid_y, grid_z = adaptive_contour(tmp_chi2fn, x_1D = run_settings["contour_xs"],
                                                           y_1D = run_settings["contour_ys"],
                                                           contour_levels = np.sort(np.array([2.29575, 6.18007, 11.8292] + [1.0, 5.99146]*model.count("w0wa"))), max_depth = max_depth,
                                                           use_parallel = use_parallel) # Nominal max_depth = 5!

        
        all_grids[the_name] = (grid_x, grid_y, grid_z)
        

    
        plt.figure(figsize = (48, 48))
        
        plt.scatter(all_xyz[0], all_xyz[1], c = all_xyz[2], cmap = "nipy_spectral", vmin = 0, vmax = 50)
        for i in range(len(all_xyz[0])):
            plt.text(all_xyz[0][i], all_xyz[1][i], "%.2g" % all_xyz[2][i], size = 2, rotation = 20)
        plt.colorbar()
        
        plt.savefig("adapt_" + the_name + "_" + model + ".pdf")
        plt.close()


    pickle.dump(all_grids, open(get_output_pickle(model), 'wb'))


def get_output_pickle(model):
    return "all_grids_" + model + "_" + SN_matrix.split(".fits")[0].split("/")[-1] + "_max=" + str(max_depth) + ".pickle"

    

print("python compute_chi2s.py mu_mat.fits 4")


SN_matrix = sys.argv[1]
max_depth = int(sys.argv[2])


use_parallel = int(sys.argv[3])
# For big SN_matrices (i.e., unbinned), parallelizing numpy is faster than parallelizing chi^2 calls, so want use_parallel = 0. Maybe also for Arm64.

try:
    models_to_run = sys.argv[4:]
except:
    models_to_run = ["flatw0wa", "flatwCDM", "flatLCDM", "w0wa", "LCDM"]

print("models_to_run", models_to_run)

f = fits.open(SN_matrix)
dat = f[0].data
f.close()

mu_list = dat[1:, 0]

if np.max(np.abs(mu_list)) < 10:
    print("This looks like LCDM residuals! Adding mu(z)")

    mu_list += get_mu(z_list = dat[0, 1:],
                      cosmo = dict(model = "flatLCDM", O_m = 0.3, O_k = 0.0, h = 0.7))

    dat[1:, 0] = mu_list
    save_img(dat, SN_matrix.replace(".fits", "_mu.fits"))
    
z_list = dat[0, 1:]
mu_invcov = dat[1:, 1:]

"""
if SN_matrix.count("P+"):
    pass
else:
    print("BLINDED!!!!!"*100)
    mu_list = get_mu(z_list, cosmo = dict(model = "flatLCDM", O_bhh = 0.022, h = 0.67, O_m = 0.31, O_k = 0.))
"""


print("use_parallel", use_parallel)

BAO_data = load_BAO()

if models_to_run.count("binnedrho"):
    all_results = {}
    for include_BAO in [1]:
        for zbins in tqdm.tqdm([[0.2, 0.5, 1.0, 2.0],
                                [0.2, 0.5, 1.0, 1.5, 2.0],
                                [0.1, 0.2, 0.5, 1.0, 1.5, 2.0],
                                [0.2, 0.5, 2.0, 4.0],
                                [0.5, 1.0, 2.0],
                                [0.5, 1.0, 1.6],
                                [0.5, 1.0, 2.2],
                                [0.5, 2.0, 4.0],
                                [0.2, 0.5, 1.0, 2.0, 4.0]]):
            all_results[(include_BAO, zbins)] = binned_constraints(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, zbins = zbins, include_BAO = include_BAO)
    pickle.dump(all_results, open(get_output_pickle("binnedrho"), 'wb'))
    
    del models_to_run[models_to_run.index("binnedrho")]
    
for model_to_run in models_to_run:
    make_contours(z_list = z_list, mu_list = mu_list, mu_invcov = mu_invcov, model = model_to_run)#"flatwCDM")#"LCDM")

