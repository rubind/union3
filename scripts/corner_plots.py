import numpy as np
import pickle as pickle
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile, percentileofscore
import gzip
import sys
from kde_corner import kde_corner

def get_label(key):
    try:
        return {"H0": "$H_0$",
                "MB:0": "$\mathcal{M}_B$",
                "alpha": "$\\alpha$",
                "beta_B": "$\\beta_B$",
                "beta_R": "$\\beta_R$",
                "delta_beta_R": "$\Delta \\beta_R \equiv \\beta_{RH} - \\beta_{RL}$",
                "Om": "$\Omega_m$",
                "mean_sigma_int": "$<\sigma^{\mathrm{unexpl.}}>$",
                "outl_frac": "$f^{\mathrm{outl}}$",
                "delta_0": "$\delta(z=0)$",
                "outl_mBx1c_uncertainties_mB": "$\sigma^{\mathrm{outl}}_{m_B}$",
                "outl_mBx1c_uncertainties_x1": "$\sigma^{\mathrm{outl}}_{x_1}$",
                "outl_mBx1c_uncertainties_cB": "$\sigma^{\mathrm{outl}}_{c_B}$",
                "outl_mBx1c_uncertainties_cR_unit": "$\sigma^{\mathrm{outl}}_{c_R}/\\tau^c$",
                "delta_h": "$\delta(z=\infty)/\delta(z=0)$"}[key]
    except:
        pass

    if key.count(":"):
    
        if key.count("mBx1c_int_variance:"):
            mBx1c = ["{m_B}", "{x_1}", "c"][int(key.split(":")[-1])]
            return "$f^%s$" % mBx1c
        
        if key.count("outl_mBx1c_uncertainties:"):
            mBx1c = ["{m_B}", "{x_1}", "{c_B}", "{c_R}"][int(key.split(":")[-1])]
            return "$\sigma^{\mathrm{outl}}_%s$" % mBx1c


    return key


def make_plot(keys, pltname):
    labels = []
    samples = []


    for key in keys:

        if key.count(":") == 0:
            samples.append(fit_params[key])
        else:
            ind = int(key.split(":")[-1])
            samples.append(fit_params[key.split(":")[0]][:,ind])
            print("taking ", ind, key)

        labels.append(get_label(key))


    kde_corner(samples, labels, bw_method = 0.2, colors = [[(104/255., 140/255., 184/255.), (37/255., 85/255., 145/255.)]], ax_limits = [(item.min(), item.max()) for item in samples], labelfontsize = 14)
    plt.savefig(pltname, bbox_inches = 'tight')
    plt.close()


plt_choice = int(sys.argv[1]) # 0, 1, 2

pfls = sys.argv[2:]

fit_params_list = []

for pfl in pfls:
    try:
        fit_params = pickle.load(open(pfl, 'rb'))
    except:
        fit_params = pickle.load(gzip.open(pfl, 'rb'))

    for key in fit_params:
        print("fit_params", key, fit_params[key].shape)

    
    """
    for thresh in [0, 0.25, 0.5, 0.75, 1.0]:
    inds = np.where(fit_params["outl_mBx1c_uncertainties"][:, 3] > thresh)
    print(len(inds[0]))
    print(thresh, "Om", np.median(fit_params["Om"][inds]))
    
    fdlksjfljk
    """ 


    fit_params["beta_R"] = 0.5*(fit_params["beta_R_high"] + fit_params["beta_R_low"])
    fit_params["delta_beta_R"] = fit_params["beta_R_high"] - fit_params["beta_R_low"]
    fit_params["mean_sigma_int"] = np.mean(fit_params["sigma_int"], axis = 1)

    fit_params_list.append(fit_params)

    has_H0 = 0
    if "H0" in fit_params:
        if np.std(fit_params["H0"]) < 6:
            has_H0 = 1
        

    
if plt_choice == 0:
    
    make_plot(["H0"]*has_H0 + ["Om", "alpha", "beta_B", "beta_R", "delta_beta_R", "MB:0", "delta_0", "delta_h"], "standardization_coeffs.pdf")
elif plt_choice == 1:
    try:
        fit_params["outl_mBx1c_uncertainties_mB"]
        indiv_labels = 1
    except:
        indiv_labels = 0

    make_plot(["H0"]*has_H0 + ["Om", "mean_sigma_int"] + ["sigma_int_calibrator"]*has_H0 + ["mBx1c_int_variance:0", "mBx1c_int_variance:1", "mBx1c_int_variance:2", "outl_frac"] + (indiv_labels == 0)*["outl_mBx1c_uncertainties:0", "outl_mBx1c_uncertainties:1", "outl_mBx1c_uncertainties:2", "outl_mBx1c_uncertainties:3"] + indiv_labels*["outl_mBx1c_uncertainties_mB", "outl_mBx1c_uncertainties_x1", "outl_mBx1c_uncertainties_cB", "outl_mBx1c_uncertainties_cR_unit"], "uncertainty_parameters.pdf")
elif plt_choice == 2:
    make_plot(["H0"]*has_H0 + ["Om", "alpha", "beta_B", "beta_R", "delta_beta_R", "mean_sigma_int"] + ["sigma_int_calibrator"]*has_H0 + ["mBx1c_int_variance:0", "mBx1c_int_variance:1", "mBx1c_int_variance:2"], "standardization_unexplained.pdf")
