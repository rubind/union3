import glob
try:
    from commands import getoutput
except:
    from subprocess import getoutput

import numpy as np
from FileRead import read_param
import sys

def fmt(val, unc, mean_unc):
    if np.isnan(val):
        assert np.isnan(unc)
        return "\\nodata"
    else:
        return "$%.3f \pm %.3f \pm %.3f$" % (val, unc, mean_unc)

    

try:
    suffix = sys.argv[1]
except:
    print("Needs suffix like LH or H")
    assert 0
    
if suffix == "H":
    cosmomodel = "1"
else:
    cosmomodel = "5"

pars = ["Om"]*(cosmomodel == 1) + ["wDE", "waDE"]*(cosmomodel == 5) + ["alpha", "beta_B", "beta_R_low", "beta_R_high", "delta_0", "delta_h", "mobs_cuts\[1\]", "mobs_cut_sigmas\[1\]", "sigma_int\[1\]", "mBx1c_int_variance\[1\]", "mBx1c_int_variance\[2\]", "mBx1c_int_variance\[3\]"]


labels = {"Om": "$\Omega_m$", "wDE": "$w_0$", "waDE": "$w_a$", "this_MB": "$\mathcal{M}_B$", "alpha": "$\\alpha$",
          "beta_B": "$\\beta_B$",
          "beta_R_low": "$\\beta_{RL}$",
          "beta_R_high": "$\\beta_{RH}$",
          "delta_0": "$\delta(0)$",
          "delta_h": "$\delta(\infty)/\delta(0)$",
          "mobs_cuts\[1\]": "$m_{50}$",
          "mobs_cut_sigmas\[1\]": "$\sigma_m$",
          "sigma_int\[1\]": "$\sigma^{\mathrm{unexpl}}$",
          "mBx1c_int_variance\[1\]": "$f^{m_B}$",
          "mBx1c_int_variance\[2\]": "$f^{x_1}$",
          "mBx1c_int_variance\[3\]": "$f^{c}$"}


"""
for x1c	in ["x1", "c"]:
    for	x1cfmt in ["%s_star", "R_%s", "tau_%s"]:
        for i in range(2):
            LH = "HL"[i]
            the_par = (x1cfmt % x1c) + ("\[%s\]" % (i+1))
            pars.append(the_par)
            if x1cfmt.count("star"):
                labels[the_par] = "$" + x1c.replace("x1", "x") + "_{%s%s}^{*}$" % ("1"*(x1c == "x1"), LH)
            if x1cfmt.count("R"):
                labels[the_par] = "$R^{%s}_%s$" % (x1c.replace("x1", "x_1"), LH)
            if x1cfmt.count("tau"):
                labels[the_par] = "$\\tau_{%s%s}$" % (x1c.replace("x1", "x_1"), LH)
"""

all_txt_grid = []

for matchstr, description in [
        ("UNITY" + suffix + "_nosel_cos=" + cosmomodel + "_???/log.txt", "No Selection Effects"),
        ("UNITY" + suffix + "_nosel_twopop_cos=" + cosmomodel + "_???/log.txt", "No Selection Effects"),
        ("UNITY" + suffix + "_cos=" + cosmomodel + "_???/log.txt", "Nominal UNITY1.5 Model")]:
        #("UNITY" + suffix + "_1D_???/log.txt", "UNITY1.5, 1D Unexplained"),
        #("UNITY" + suffix + "_nocal_???/log.txt", "UNITY1.5, No $\Delta$sys")

    logs = glob.glob(matchstr)
    

    all_pars = {}
    all_uncs = {}

    for par in pars:
        all_pars[par] = []
        all_uncs[par] = []

        for log in logs:
            val = getoutput("grep '" + par + " ' " + log)
            try:
                themean = float(val.split(None)[1])
                theunc = float(val.split(None)[3])
                all_pars[par].append(themean)
                all_uncs[par].append(theunc)
                
            except:
                print("SKIPPING", val, log, par)

            if ["mBx1c_int_variance\[1\]", "mBx1c_int_variance\[2\]", "mBx1c_int_variance\[3\]"].count(par):
                if read_param(log.split("/")[0] + "/paramfile.txt", "threeD_unexplained") == 0:
                    if par == "mBx1c_int_variance\[1\]":
                        all_pars[par][-1] = 1
                        all_uncs[par][-1] = 0
                    else:
                        all_pars[par][-1] = 0
                        all_uncs[par][-1] = 0                        
            if par.count("mobs_cut"):
                if read_param(log.split("/")[0] + "/paramfile.txt", "stan_code").count("no_sel"):
                    all_pars[par][-1] = np.sqrt(-1)
                    all_uncs[par][-1] = np.sqrt(-1)
                        
    print("all_pars", all_pars, len(all_pars["Om"]))
    towrite = [description]
    for par in pars:
        the_mean = np.mean(all_pars[par])
        the_std = np.std(all_pars[par], ddof=1)
        sqrtn = np.sqrt(float(len(all_pars[par])))

        mean_unc = np.mean(all_uncs[par])
        
        towrite.append(fmt(the_mean, the_std/sqrtn, mean_unc))

        #the_mean = np.mean(all_uncs[par])
        #the_std = np.std(all_uncs[par], ddof=1)

        #towrite.append(fmt(the_mean, the_std/sqrtn))
        
    all_txt_grid.append(towrite)
    

all_txt_grid = np.array(all_txt_grid)    

print("all_txt_grid", all_txt_grid, all_txt_grid.shape)


print("Parameter & " + " & ".join(all_txt_grid[:,0]))
for i in range(len(pars)):
    for valunc in range(1):
        towrite = labels[pars[i]] + " & "
        
        for j in range(len(all_txt_grid)):
            towrite += all_txt_grid[j][i+1+valunc]
        
            if j != len(all_txt_grid) -1:
                towrite += " & "
            
        towrite += "\\\\"
        print(towrite)
    
