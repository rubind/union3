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
        return {"MB:0": "$\mathcal{M}_B$",
                "alpha": "$\\alpha$",
                "beta_B": "$\\beta_B$",
                "beta_R": "$\\beta_R$",
                "delta_beta_R": "$\Delta \\beta_R \equiv \\beta_{RH} - \\beta_{RL}$",
                "Om": "$\Omega_m$",
                "mean_sigma_int": "$<\sigma^{\mathrm{unexpl.}}>$",
                "outl_frac": "$f^{\mathrm{outl}}$",
                "delta_0": "$\delta(z=0)$",
                "delta_h": "$\delta(z=\infty)/\delta(z=0)$"}[key]
    except:
        pass

    if key.count(":"):
        mBx1c = ["{m_B}", "{x_1}", "c"][int(key.split(":")[-1])]
    
        if key.count("mBx1c_int_variance:"):
            return "$f^%s$" % mBx1c
        if key.count("outl_mBx1c_uncertainties"):
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


    kde_corner(samples, labels, bw_method = 0.2, colors = [(104/255., 140/255., 184/255.), (37/255., 85/255., 145/255.)], ax_limits = [(item.min(), item.max()) for item in samples], labelfontsize = 14)
    plt.savefig(pltname, bbox_inches = 'tight')
    plt.close()


try:
    fit_params = pickle.load(open(sys.argv[1], 'rb'))
except:
    fit_params = pickle.load(gzip.open(sys.argv[1], 'rb'))

for key in fit_params:
    print("fit_params", key, fit_params[key].shape)
    

fit_params["beta_R"] = 0.5*(fit_params["beta_R_high"] + fit_params["beta_R_low"])
fit_params["delta_beta_R"] = fit_params["beta_R_high"] - fit_params["beta_R_low"]
fit_params["mean_sigma_int"] = np.mean(fit_params["sigma_int"], axis = 1)


make_plot(["Om", "alpha", "beta_B", "beta_R", "delta_beta_R", "MB:0", "delta_0", "delta_h"], "standardization_coeffs.pdf")
make_plot(["Om", "mean_sigma_int", "mBx1c_int_variance:0", "mBx1c_int_variance:1", "mBx1c_int_variance:2", "outl_frac", "outl_mBx1c_uncertainties:0", "outl_mBx1c_uncertainties:1", "outl_mBx1c_uncertainties:2"], "uncertainty_parameters.pdf")

