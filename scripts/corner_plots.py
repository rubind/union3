import numpy as np
import pickle as pickle
from matplotlib import use
use("PDF")
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile, percentileofscore
import gzip
import sys
from kde_corner import kde_corner

def fmt_vals(vals):
    percentiles = scoreatpercentile(vals, [15.8655, 50., 84.1345])
    
    smallest_unc = (percentiles[1:] - percentiles[:-1]).min()
    decimal_places = int(np.around(1.5 - np.log10(smallest_unc))) #int(np.around(0.80102999566 - np.log10(smallest_unc)))

    fmt_txt = "%." + str(decimal_places) + "f"

    fmt_plus = fmt_txt % (percentiles[2] - percentiles[1])
    fmt_minus = fmt_txt % (percentiles[1] - percentiles[0])

    if fmt_plus != fmt_minus:
        return (fmt_txt + "^{+%s}_{-%s}") % (percentiles[1], fmt_plus, fmt_minus)
    else:
        return (fmt_txt + "\pm %s") % (percentiles[1], fmt_plus)
                

def get_label(key):
    try:
        return {"H0": "$H_0$",
                "MB:0": "$\mathcal{M}_B$",
                "MB_slow:0": "$\mathcal{M}_B$ slow",
                "MB_fast_minus_slow": "$\mathcal{M}_B$ fast - slow",
                "alpha": "$\\alpha$",
                "beta_B": "$\\beta_B$",
                "beta_R": "$\\beta_R$",
                "delta_beta_R": "$\Delta \\beta_R(z=0) \equiv \\beta_{RH}(z=0) - \\beta_{RL}(z=0)$",
                "Om": "$\Omega_m$",
                "mean_sigma_int": "$<\sigma^{\mathrm{unexpl.}}>$",
                "outl_frac": "$f^{\mathrm{outl}}$",
                "delta_0": "$\delta(z=0)$",
                "outl_mBx1c_uncertainties_mB": "$\sigma^{\mathrm{outl}}_{m_B}$",
                "outl_mBx1c_uncertainties_x1": "$\sigma^{\mathrm{outl}}_{x_1}$",
                "outl_mBx1c_uncertainties_cB": "$\sigma^{\mathrm{outl}}_{c_B}$",
                "outl_mBx1c_uncertainties_cR_unit": "$\sigma^{\mathrm{outl}}_{c_R}/\\tau^c$",
                "delta_h": "$\delta(z=\infty)/\delta(z=0)$\n\Delta \\beta_R(z=\infty)/\Delta \\beta_R(z=0)"}[key]
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

def get_samples_labels(keys):
    labels = []
    samples = []


    for key in keys:
        if key.split(":")[0] in fit_params:
            if key.count(":") == 0:
                if np.std(fit_params[key]) > 0:
                    samples.append(fit_params[key])
                    labels.append(get_label(key))
            else:
                ind = int(key.split(":")[-1])
                if np.std(fit_params[key.split(":")[0]][:,ind]) > 0:
                    samples.append(fit_params[key.split(":")[0]][:,ind])
                    print("taking ", ind, key)
                    labels.append(get_label(key))
    return samples, labels


def find_large_coeff():
    samples = []
    labels = []
    ignore = ["beta_eff", "p_high_mass", "p_high_mass_eff", "this_delta_h", "this_MB_slow", "this_norm_LL_fast", "this_norm_LL_slow", "beta_eff", "beta_angle_blue", "alpha_angle_fast", "alpha_angle_slow", "beta_angle_red_low", "beta_angle_red_high", "tmploglike_c", "lp__", "lp__1000"]
    
    for key in fit_params:
        if ignore.count(key) == 0:
            the_shape = fit_params[key].shape
            if len(the_shape) == 1:
                labels.append(get_label(key))
                samples.append(fit_params[key])
            else:
                if len(the_shape) == 2:
                    if the_shape[1] < 10:
                        for i in range(the_shape[1]):
                            labels.append(key + ":" + str(i))
                            samples.append(fit_params[key][:, i])
    
                        
    samples = np.array(samples)
    print("samples", samples.shape)
    corr = np.corrcoef(samples)
    print("corr", corr.shape)

    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if np.abs(corr[i,j]) > 0.5:
                print(labels[i], labels[j], corr[i,j])
    lfkdalfkdj

def make_plot(keys, pltname):
    samples, labels = get_samples_labels(keys)

    kde_corner(samples, labels, bw_method = 0.2, colors = [[(104/255., 140/255., 184/255.), (37/255., 85/255., 145/255.)]], ax_limits = [(item.min(), item.max()) for item in samples], labelfontsize = 14)
    plt.savefig(pltname, bbox_inches = 'tight')
    plt.close()

def make_latex_table(fit_params):
    params = ["alpha", "alpha_fast", "alpha_slow", "delta_alpha", "beta_B", "beta_R", "delta_beta_R", "beta_R_high", "beta_R_low", "MB_slow", "MB_fast_minus_slow"] + ["step_mass", "mass_width"] + ["delta_0", "delta_h"] + ["R_x1_fast", "R_x1_slow", "x1_star_fast", "x1_star_slow"] + ["R_c_fast", "R_c_slow", "c_star_fast", "c_star_slow"] + ["outl_frac", "sigma_int_fast", "mean_sigma_int", "mBx1c_int_variance[1]", "mBx1c_int_variance[2]", "mBx1c_int_variance[3]"]

    f = open("fit_params.tex", 'w')
    
    for param in params:
        val_from_fit_params = []
        
        if param in fit_params:
            val_from_fit_params = fit_params[param]
        elif param.split("[")[0] in fit_params:
            assert param[-1] == "]", param
            assert param[-3] == "[", param
            
            val_from_fit_params = fit_params[param.split("[")[0]][:, int(param[-2]) - 1]
            
        if len(val_from_fit_params) > 2:
            f.write("\\newcommand{\\" + param.replace("_", "").replace(":", "").replace("0", "zero").replace("1", "one").replace("2", "two").replace("3", "three").replace("[", "").replace("]", "") + "value}{\ensuremath{" + fmt_vals(val_from_fit_params) + "}\\xspace}\n")

    for i in range(len(fit_params["tau_c"][0])):
        for param in ["tau_c", "frac_x1_slow"]:
            if param in fit_params:
                suffix = ["lowz", "midz", "highz"][i % 3] + "highmass"*(i <= 2) + "lowmass"*(i > 2)
                f.write("\\newcommand{\\" + param.replace("_", "").replace(":", "").replace("0", "zero").replace("1", "one") + suffix + "value}{\ensuremath{" + fmt_vals(fit_params[param][:,i]) + "}\\xspace}\n")
        
    f.close()
    print("Done with latex commands")
    
    


plt_choice = int(sys.argv[1]) # 0, 1, 2

pfls = sys.argv[2:]

fit_params_list = []

for pfl in pfls:
    try:
        fit_params = pickle.load(open(pfl, 'rb'))
    except:
        fit_params = pickle.load(gzip.open(pfl, 'rb'))


    for key in fit_params:
        fit_params[key] = fit_params[key]
        print("fit_params", key, fit_params[key].shape, np.median(fit_params[key]), np.std(fit_params[key], ddof=1))

    if "alpha_fast" in fit_params and "alpha_slow" in fit_params:
        fit_params["delta_alpha"] = fit_params["alpha_fast"] - fit_params["alpha_slow"]

        
    
    """
    for thresh in [0, 0.25, 0.5, 0.75, 1.0]:
    inds = np.where(fit_params["outl_mBx1c_uncertainties"][:, 3] > thresh)
    print(len(inds[0]))
    print(thresh, "Om", np.median(fit_params["Om"][inds]))
    
    fdlksjfljk
    """ 

    try:
        fit_params["beta_R"] = 0.5*(fit_params["beta_R_high"] + fit_params["beta_R_low"])
        fit_params["delta_beta_R"] = fit_params["beta_R_high"] - fit_params["beta_R_low"]
    except:
        fit_params["beta_R"] = 0.5*(fit_params["beta_R_slow"] + fit_params["beta_R_fast"])
        fit_params["delta_beta_R"] = fit_params["beta_R_slow"] - fit_params["beta_R_fast"]

        
    fit_params["mean_sigma_int"] = np.mean(fit_params["sigma_int"], axis = 1)
    
    make_latex_table(fit_params)

    fit_params_list.append(fit_params)

    has_H0 = 0
    if "H0" in fit_params:
        if np.std(fit_params["H0"]) < 6:
            has_H0 = 1
        


if "frac_x1_slow" in fit_params:
    plt.figure(figsize = (12, 12))
    
    for i in range(len(fit_params["frac_x1_slow"][0])):
        plt.subplot(3,3,i+1)
        plt.hist(fit_params["frac_x1_slow"][:,i], bins = 30)
    plt.savefig("frac_x1_slow.pdf", bbox_inches = 'tight')
    plt.close()


fit_params["lp__1000"] = fit_params["lp__"]/1000.

if plt_choice == 0:
    
    make_plot(["H0"]*has_H0 + ["Om", "alpha", "alpha_fast", "alpha_slow", "beta_B", "beta_R", "delta_beta_R", "MB_slow:0", "MB_fast_minus_slow"] + ["step_mass", "mass_width"] + ["delta_0", "delta_h"] + ["lp__1000"]*0, "standardization_coeffs.pdf")
elif plt_choice == 1:
    try:
        fit_params["outl_mBx1c_uncertainties_mB"]
        indiv_labels = 1
    except:
        indiv_labels = 0

    make_plot(["H0"]*has_H0 + ["Om", "mean_sigma_int"] + ["sigma_int_calibrator"]*has_H0 + ["mBx1c_int_variance:0", "mBx1c_int_variance:1", "mBx1c_int_variance:2", "outl_frac"] + (indiv_labels == 0)*["outl_mBx1c_uncertainties:0", "outl_mBx1c_uncertainties:1", "outl_mBx1c_uncertainties:2", "outl_mBx1c_uncertainties:3"] + indiv_labels*["outl_mBx1c_uncertainties_mB", "outl_mBx1c_uncertainties_x1", "outl_mBx1c_uncertainties_cB", "outl_mBx1c_uncertainties_cR_unit"], "uncertainty_parameters.pdf")
elif plt_choice == 2:
    make_plot(["H0"]*has_H0 + ["Om", "alpha", "beta_B", "beta_R", "delta_beta_R", "mean_sigma_int", "sigma_int_fast"] + ["delta_0"]*0 + ["sigma_int_calibrator"]*has_H0 + ["mBx1c_int_variance:0", "mBx1c_int_variance:1", "mBx1c_int_variance:2"], "standardization_unexplained.pdf")
elif plt_choice == 3:
    find_large_coeff()
    
